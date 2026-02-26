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
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

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

        return jnp.concatenate([h_collision, h_singularity, h_table])

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
    return cbf.safety_filter(z, desired_joint_torque)


class OSCBFNode(Node):
    def __init__(self):
        super().__init__("oscbf_franka_forward_torque_node")
        self.get_logger().info("Starting Franka Forward Torque Node")

        qos_profile = QoSProfile(
            depth=1,                                  # <-- reduced from 10: no point buffering stale states
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.torque_cmd_pub = self.create_publisher(
            Float64MultiArray, "/forward_torque_controller/commands", qos_profile
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, qos_profile
        )
        self.desired_torque_sub = self.create_subscription(
            Float64MultiArray, "/desired_joint_torque", self.desired_torque_callback, qos_profile
        )

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.get_logger().info("Loading Franka Model...")
        self.robot = load_panda()
        nj = self.robot.num_joints

        # Pre-allocate everything — zero heap allocation in the hot loop
        self.last_joint_state = np.zeros(nj * 2)
        self.desired_joint_torque = np.zeros(nj)
        self.got_joint_state = False
        self.got_desired_torque = False
        self.cmd_msg = Float64MultiArray()
        self.cmd_msg.data = [0.0] * nj        # pre-allocate the list once

        self.get_logger().info("Initializing CBF...")
        z_min = 0.0
        num_bodies = 8
        x1, y1 = 0.45, 0.32
        x2, y2 = 0.36, 0.176
        all_collision_pos = np.array([
            [x1, y1, 0.03], [x1, y1, 0.09], [x1, y1, 0.15], [x1, y1, 0.21],
            [x2, y2, 0.03], [x2, y2, 0.09], [x2, y2, 0.15], [x2, y2, 0.21],
        ])
        # all_collision_pos = np.array([[x1, y1, 0.03]])
        all_collision_radii = np.repeat(0.03, len(all_collision_pos))
        collision_pos = np.atleast_2d(all_collision_pos[:num_bodies])
        collision_radii = all_collision_radii[:num_bodies]

        self.cbf_config = FrankaConfig(self.robot, z_min, collision_pos, collision_radii)
        self.cbf = CBF.from_config(self.cbf_config)

        self.get_logger().info("JIT compiling OSCBF controller...")
        self._jit_compile()

        self.control_freq = 1000.0
        self.timer = self.create_timer(1.0 / self.control_freq, self.publish_control)
        self.get_logger().info("Franka Forward Torque Node Initialized")

    def _jit_compile(self):
        z = np.zeros(self.robot.num_joints * 2)
        desired_joint_torque = np.zeros(self.robot.num_joints)
        _ = np.asarray(compute_control(self.robot, self.cbf, z, desired_joint_torque))
        _ = np.asarray(compute_control(self.robot, self.cbf, z, desired_joint_torque))  # second run to warm caches

    def joint_state_callback(self, msg: JointState):
        nj = self.robot.num_joints
        # Write in-place: no allocation
        self.last_joint_state[:nj] = msg.position[:nj]
        self.last_joint_state[nj:] = msg.velocity[:nj]
        self.got_joint_state = True

    def desired_torque_callback(self, msg: Float64MultiArray):
        # Write in-place: no allocation
        self.desired_joint_torque[:] = msg.data[:self.robot.num_joints]
        self.got_desired_torque = True

    def publish_control(self):
        if not self.got_joint_state or not self.got_desired_torque:
            return

        tau = np.asarray(compute_control(
            self.robot,
            self.cbf,
            self.last_joint_state,
            self.desired_joint_torque,
        ))

        # Write into the pre-allocated list — no new list/array created
        for i in range(self.robot.num_joints):
            self.cmd_msg.data[i] = float(tau[i])

        self.torque_cmd_pub.publish(self.cmd_msg)

    def signal_handler(self, sig, frame):
        self.get_logger().warn("Shutdown signal received, sending zero torques...")
        self.publish_zero_torque()
        self.get_logger().warn("Zero torques sent, shutting down.")
        time.sleep(0.1)
        sys.exit(0)

    def publish_zero_torque(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * self.robot.num_joints
        for _ in range(3):
            self.torque_cmd_pub.publish(msg)
            time.sleep(1.0 / self.control_freq)


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