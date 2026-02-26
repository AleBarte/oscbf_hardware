import signal
import time
import sys
from functools import partial
import time

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
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.oscbf_configs import OSCBFTorqueConfig

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_static
class FrankaConfig(OSCBFTorqueConfig):
    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
        singularity_tol: float = 1e-2,

    ):

        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        self.singularity_tol = singularity_tol
        super().__init__(robot)

    def h_2(self, z, *args, **kwargs):
        # Extract Values
        q = z[: self.num_joints]

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas - self.singularity_tol)])

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]

        center_deltas = (
            robot_collision_positions[:, None, :]
            - self.collision_positions[None, :, :]
        ).reshape(-1, 3)

        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)

        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole body table avoidance
        h_table = (
            robot_collision_positions[:, 2] - robot_collision_radii.ravel() - self.z_min
        )

        return jnp.concatenate([h_singularity, h_collision, h_table])
    
    def alpha(self, h):
        return 10.0 * h
    
    def alpha_2(self, h_2):
        return 10.0 * h_2
    
@partial(jax.jit, static_argnums=(0, 1))
def compute_control(
    robot: Manipulator,
    cbf: CBF,
    z: ArrayLike,
    desired_joint_torque: ArrayLike,
):
    # Apply CBF Safety Filter
    tau = cbf.safety_filter(z, desired_joint_torque)
    # TODO: Understand if we have to also account for gravity like they do in their node
    return tau


class OSCBFNode(Node):
    def __init__(
            self,
    ):
        super().__init__("oscbf_franka_forward_torque_node")
        self.get_logger().info("Starting Franka Forward Torque Node")

        qos_profile = QoSProfile(
            depth = 10,
            reliability = ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.torque_cmd_pub = self.create_publisher(Float64MultiArray, "/forward_torque_controller/commands", qos_profile)
        self.marker_pub = self.create_publisher(MarkerArray, "/oscbf/collision_objects", 10)
        self.joint_state_sub = self.create_subscription(JointState, "/joint_states", self.joint_state_callback, qos_profile)
        # self.joint_state_initialization_sub = self.create_subscription(JointState, "/joint_states", self.joint_state_initialization_callback, qos_profile)
        self.desired_torque_sub = self.create_subscription(Float64MultiArray, "/desired_joint_torque", self.desired_torque_callback, qos_profile)

        # self.got_orderer_joint_state = False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.control_freq = 1000.0
        self.timer = self.create_timer(1.0 / self.control_freq, self.publish_control)

        # Initialize
        self.last_torque_cmd = None
        self.last_joint_state = None
        self.desired_joint_torque = None
        # self.ordered_joint_indexes = []

        self.get_logger().info("Loading Franka Model...")
        self.robot = load_panda()
        self.sorted_positions = np.zeros(self.robot.num_joints)
        self.sorted_velocities = np.zeros(self.robot.num_joints)

        self.get_logger().info("Initializing CBF...")

        z_min = 0.0
        num_bodies = 8

        x1 = 0.45
        y1 = 0.32
        x2 = 0.36
        y2 = 0.176
        all_collision_pos = np.array([[x1, y1, 0.03],
                                      [x1, y1, 0.09],
                                      [x1, y1, 0.15],
                                      [x1, y1, 0.21],
                                      [x2, y2, 0.03],
                                      [x2, y2, 0.09],
                                      [x2, y2, 0.15],
                                      [x2, y2, 0.21]])
        # all_collision_radii = np.random.uniform(low=0.01, high=0.1, size=(max_num_bodies,))
        all_collision_radii = np.repeat(0.03, len(all_collision_pos))
        # Only use a subset of them based on the desired quantity
        collision_pos = np.atleast_2d(all_collision_pos[:num_bodies])
        collision_radii = all_collision_radii[:num_bodies]

        self.collision_positions = collision_pos
        self.collision_radii = collision_radii

        self.cbf_config = FrankaConfig(self.robot, z_min, collision_pos, collision_radii)
        self.cbf = CBF.from_config(self.cbf_config)

        self.get_logger().info("Jit Compiling OSCBF controller...")
        self._jit_compile()

        # self.create_timer(1.0 / 5.0, self.publish_collision_markers)

        self.get_logger().info("Franka Forward Torque Node Initialized")
    
    def _jit_compile(self):
        # Dummy Values for Joint State and desired torques
        z = np.zeros(self.robot.num_joints * 2)
        print(self.robot.num_joints)
        desired_joint_torque = np.zeros(7)

        # Run an initial solve to compile
        _ = np.asarray(
            compute_control(self.robot, self.cbf, z, desired_joint_torque)
        )

    # def joint_state_initialization_callback(self, msg: JointState):
    #     if self.got_orderer_joint_state:
    #         return
        
    #     joint_order = [f"fer_joint1", "fer_joint2", "fer_joint3", "fer_joint4", "fer_joint5", "fer_joint6", "fer_joint7"]

    #     for joint_name in joint_order:
    #         msg_idx = msg.name.index(joint_name)
    #         self.ordered_joint_indexes.append(msg_idx)
        
    #     self.got_orderer_joint_state = True
    #     self.get_logger().info(f"Got ordered joint state indexes: {self.ordered_joint_indexes}")


    def joint_state_callback(self, msg: JointState):
        # if not self.got_orderer_joint_state:
        #     return
        
        # Sort the message data according to the expected joint order
        # for i in range(7):
        #     self.sorted_positions[i] = msg.position[self.ordered_joint_indexes[i]]
        #     self.sorted_velocities[i] = msg.velocity[self.ordered_joint_indexes[i]]

        # self.last_joint_state = np.array([self.sorted_positions, self.sorted_velocities]).ravel()

        self.last_joint_state = np.array([msg.position[:7], msg.velocity[:7]]).ravel()

    def desired_torque_callback(self, msg: Float64MultiArray):
        self.desired_joint_torque = np.array(msg.data)

    def publish_control(self):
        if self.last_joint_state is None or self.desired_joint_torque is None:
            return
        
        msg = Float64MultiArray()
        tau = compute_control(
            self.robot,
            self.cbf,
            self.last_joint_state,
            self.desired_joint_torque,
        )

        msg.data = tau.tolist()
        self.torque_cmd_pub.publish(msg)

    def signal_handler(self, sig, frame):
        """Handle shutdown signals by publishing zero torques."""
        # TODO: Decide if we should first slow down the robot to zero velocity before shutdown
        self.get_logger().warn("Shutdown signal received, sending zero torques...")
        self.publish_zero_torque()
        self.get_logger().warn("Zero torques sent, shutting down.")
        # Allow a brief moment for the message to be published
        time.sleep(0.1)
        sys.exit(0)

    def publish_zero_torque(self):
        """Publish zero torques to the robot."""
        msg = Float64MultiArray()
        msg.data = [0.0] * self.robot.num_joints
        # Publish multiple times to ensure delivery
        for _ in range(3):
            self.torque_cmd_pub.publish(msg)
            time.sleep(1 / self.control_freq)

    # def publish_collision_markers(self):
    #     """Publish collision bodies as visualization markers (spheres)."""
    #     if getattr(self, "collision_positions", None) is None:
    #         return
    #     marker_array = MarkerArray()
    #     for i, pos in enumerate(self.collision_positions):
    #         m = Marker()
    #         m.header.stamp = self.get_clock().now().to_msg()
    #         m.header.frame_id = "base_link"  # change to the appropriate frame if needed
    #         m.ns = "collision_bodies"
    #         m.id = int(i)
    #         m.type = Marker.SPHERE
    #         m.action = Marker.ADD
    #         m.pose.position.x = float(pos[0])
    #         m.pose.position.y = float(pos[1])
    #         m.pose.position.z = float(pos[2])
    #         m.pose.orientation.x = 0.0
    #         m.pose.orientation.y = 0.0
    #         m.pose.orientation.z = 0.0
    #         m.pose.orientation.w = 1.0
    #         r = float(self.collision_radii[i])
    #         m.scale.x = r * 2.0
    #         m.scale.y = r * 2.0
    #         m.scale.z = r * 2.0
    #         m.color.r = 1.0
    #         m.color.g = 0.0
    #         m.color.b = 0.0
    #         m.color.a = 1.0
    #         m.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
    #         marker_array.markers.append(m)
    #     self.marker_pub.publish(marker_array)



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
        node.publish_zero_torque()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()