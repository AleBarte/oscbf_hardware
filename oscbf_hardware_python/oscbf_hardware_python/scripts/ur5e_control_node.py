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
from visualization_msgs.msg import Marker, MarkerArray

from oscbf_hardware_python.utils.rotations_and_transforms import xyzw_to_rotation_numpy
from oscbf.core.manipulator import Manipulator, load_ur5e
from oscbf.core.oscbf_configs import OSCBFVelocityConfig
from oscbf.core.controllers import PoseTaskTorqueController

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_static
class UR5Config(OSCBFVelocityConfig):
    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
    ):
        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]
        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        print(robot_collision_positions)
        print(robot_collision_radii)
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole body table avoidance
        h_table = (
            robot_collision_positions[:, 2] - self.z_min - robot_collision_radii.ravel()
        )

        return jnp.concatenate([h_collision, h_table])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2

@partial(jax.jit, static_argnums=(0, 1))
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

        self.marker_pub = self.create_publisher(
            MarkerArray, "/oscbf/collision_objects", 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, qos_profile
        )

        self.desired_joint_vel_sub = self.create_subscription(
            Float64MultiArray, "/twist_to_joint_vel/commands", self.desired_joint_vel_callback, qos_profile
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

        self.get_logger().info("Loading UR5e model...")
        self.robot = load_ur5e()

        self.get_logger().info("Creating CBF...")

        z_min = 0.0
        num_bodies = 4
        max_num_bodies = 4

        # Sample a lot of collision bodies
        all_collision_pos = np.random.uniform(
            low=[0.2, -0.4, 0.1], high=[0.8, 0.4, 0.3], size=(max_num_bodies, 3)
        )
        all_collision_radii = np.random.uniform(low=0.01, high=0.1, size=(max_num_bodies,))
        # Only use a subset of them based on the desired quantity
        collision_pos = np.atleast_2d(all_collision_pos[:num_bodies])
        collision_radii = all_collision_radii[:num_bodies]

        self.collision_positions = collision_pos
        self.collision_radii = collision_radii

        self.cbf_config = UR5Config(self.robot, z_min, collision_pos, collision_radii)
        self.cbf = CBF.from_config(self.cbf_config)


        self.get_logger().info("Jit compiling OSCBF controller...")
        self._jit_compile()

        self.create_timer(1.0 / 5.0, self.publish_collision_markers)

        self.get_logger().info("OSCBF Node initialization complete.")

    def _jit_compile(self):
        # Dummy values for joint state and ee state
        z = np.zeros(self.robot.num_joints)
        desired_joint_vel = np.zeros(self.robot.num_joints)

        # Run an initial solve to compile
        _ = np.asarray(
            compute_control(self.robot, self.cbf, z, desired_joint_vel)
        )

    def joint_state_callback(self, msg: JointState):
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

        self.last_joint_state = np.array([sorted_positions]).ravel()        

    def desired_joint_vel_callback(self, msg: Float64MultiArray):
        self.desired_joint_vel = np.array(msg.data)
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


    def publish_collision_markers(self):
        """Publish collision bodies as visualization markers (spheres)."""
        if getattr(self, "collision_positions", None) is None:
            return
        marker_array = MarkerArray()
        for i, pos in enumerate(self.collision_positions):
            m = Marker()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "base_link"  # change to the appropriate frame if needed
            m.ns = "collision_bodies"
            m.id = int(i)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 1.0
            r = float(self.collision_radii[i])
            m.scale.x = r * 2.0
            m.scale.y = r * 2.0
            m.scale.z = r * 2.0
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)


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
        node.publish_zero_vel()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
