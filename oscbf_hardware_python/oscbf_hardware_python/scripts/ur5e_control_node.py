"""ROS2 Controller Node for the Franka

Publishes:
- Joint torques
Subscribes:
- Joint states (position, velocity)
- Desired end-effector state (position, orientation, velocity, angular velocity)
"""

import signal
import time
import sys
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from cbfpy import CBF

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from oscbf_msgs.msg import EEState

from oscbf_hardware_python.utils.rotations_and_transforms import xyzw_to_rotation_numpy
from oscbf.core.manipulator import Manipulator, load_ur5e
from oscbf.core.oscbf_configs import OSCBFVelocityConfig
from oscbf.core.controllers import PoseTaskTorqueController

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_static
class UR5Config(OSCBFVelocityConfig):
    """CBF Config for demoing OSCBF on the UR5e hardware

    Safety Constraints:
    - Joint limit avoidance
    - Singularity avoidance
    - Whole-body set containment
    """

    def __init__(
        self,
        robot: Manipulator,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
    ):
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-2
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    def h_2(self, z, *args, **kwargs):
        # Extract values
        q = z[: self.num_joints]
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        robot_num_pts = robot_collision_positions.shape[0]

        # Whole-body Set Containment
        h_whole_body_upper = (
            jnp.tile(self.whole_body_pos_max, (robot_num_pts, 1))
            - robot_collision_positions
            - robot_collision_radii
        ).ravel()
        h_whole_body_lower = (
            robot_collision_positions
            - jnp.tile(self.whole_body_pos_min, (robot_num_pts, 1))
            - robot_collision_radii
        ).ravel()

        return jnp.concatenate(
            [
                h_joint_limits,
                h_singularity,
                h_whole_body_upper,
                h_whole_body_lower,
            ]
        )

    def h_1(self, z, *args, **kwargs):
        qdot = z[self.num_joints :]
        # Joint velocity limits
        joint_max_vels = jnp.asarray(self.robot.joint_max_velocities)
        qdot_max = joint_max_vels
        qdot_min = -joint_max_vels
        return jnp.concatenate([qdot_max - qdot, qdot - qdot_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


@partial(jax.jit, static_argnums=(0, 1, 2))
def compute_control(
    robot: Manipulator,
    cbf: CBF,
    z: ArrayLike,
    desired_joint_vel: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]

    # Apply the CBF safety filter
    tau = cbf.safety_filter(z, desired_joint_vel)

    return tau


class OSCBFNode(Node):
    def __init__(
        self,
        whole_body_pos_min: ArrayLike = (-0.5, -0.5, 0.0),
        whole_body_pos_max: ArrayLike = (0.75, 0.5, 1.0),
    ):
        super().__init__("oscbf_node")
        self.get_logger().info("Initializing OSCBF Node...")
        whole_body_pos_min = np.asarray(whole_body_pos_min)
        whole_body_pos_max = np.asarray(whole_body_pos_max)
        assert whole_body_pos_min.shape == (3,)
        assert whole_body_pos_max.shape == (3,)

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.vel_cmd_pub = self.create_publisher(
            Float64MultiArray, "/forward_velocity_controller/commands", qos_profile
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, qos_profile
        )

        self.desired_joint_vel_sub = self.create_subscription(
            Float64MultiArray, "/", self.desired_joint_vel_callback, qos_profile
        )

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.control_freq = 500
        self.timer = self.create_timer(1 / self.control_freq, self.publish_control)

        # Initialize
        self.last_vel_cmd = None
        self.last_joint_state = None
        self.desired_joint_vel = None

        self.get_logger().info("Loading Franka model...")
        self.robot = load_ur5e()

        self.get_logger().info("Creating CBF...")
        self.cbf_config = UR5Config(self.robot, whole_body_pos_min, whole_body_pos_max)
        self.cbf = CBF.from_config(self.cbf_config)


        self.get_logger().info("Jit compiling OSCBF controller...")
        self._jit_compile()

        self.get_logger().info("OSCBF Node initialization complete.")

    def _jit_compile(self):
        # Dummy values for joint state and ee state
        z = np.zeros(self.robot.num_joints * 2)
        desired_joint_vel = np.zeros(self.robot.num_joints)

        # Run an initial solve to compile
        _ = np.asarray(
            compute_control(self.robot, self.cbf, z, desired_joint_vel)
        )

    def joint_state_callback(self, msg: JointState):
        self.last_joint_state = np.array([msg.position, msg.velocity]).ravel()

    def desired_joint_vel_callback(self, msg: Float64MultiArray):
        # Create a mapping of joint names to their indices
        joint_order = [f"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

        # Sort the message data according to the expected joint order
        sorted_positions = np.zeros(len(joint_order))
        sorted_velocities = np.zeros(len(joint_order))

        for i, joint_name in enumerate(joint_order):
            msg_idx = msg.name.index(joint_name)
            sorted_positions[i] = msg.position[msg_idx]
            sorted_velocities[i] = msg.velocity[msg_idx]

        self.last_joint_state = np.array([sorted_positions, sorted_velocities]).ravel()

    def publish_control(self):
        if self.last_joint_state is None or self.desired_joint_vel is None:
            return
        msg = Float64MultiArray()
        tau = compute_control(
            self.robot,
            self.cbf,
            self.last_joint_state,
            self.desired_joint_vel,
        )
        msg.data = tau.tolist()
        self.vel_cmd_pub.publish(msg)

    def signal_handler(self, sig, frame):
        """Handle shutdown signals by publishing zero torques."""
        # TODO: Decide if we should first slow down the robot to zero velocity before shutdown
        self.get_logger().warn("Shutdown signal received, sending zero torques...")
        self.publish_zero_vel()
        self.get_logger().warn("Zero torques sent, shutting down.")
        # Allow a brief moment for the message to be published
        time.sleep(0.1)
        sys.exit(0)

    def publish_zero_vel(self):
        """Publish zero torques to the robot."""
        msg = Float64MultiArray()
        msg.data = [0.0] * self.robot.num_joints
        # Publish multiple times to ensure delivery
        for _ in range(3):
            self.vel_cmd_pub.publish(msg)
            time.sleep(1 / self.control_freq)


def main(args=None):
    rclpy.init(args=args)
    node = OSCBFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        node.publish_zero_torques()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
