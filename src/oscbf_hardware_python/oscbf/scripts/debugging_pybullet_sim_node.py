"""Test the OSCBF ROS2 communication with pybullet

This will mimic the franka c++ communication node which listens to joint torques
and publishes joint states
"""

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

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

from oscbf.utils.rotations_and_transforms import xyzw_to_rotation_numpy
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController
from oscbf.core.manipulation_env import FrankaTorqueControlEnv


class PybulletNode(Node):
    def __init__(self):
        super().__init__("pybullet_node")

        self.freq = 1000
        self.client: pybullet = BulletClient(pybullet.GUI)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.cube_id = self.client.loadURDF(
            "/home/dmorton/ros2_ws/src/oscbf/oscbf/assets/frame.urdf",
            [0, 0, 0],
            [0, 0, 0, 1],
            globalScaling=0.5,
        )

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.ee_state_sub = self.create_subscription(
            EEState, "ee_state", self.ee_state_callback, qos_profile
        )

    def ee_state_callback(self, msg: EEState):
        pos: Point = msg.pose.position
        quat: Quaternion = msg.pose.orientation
        vel: Vector3 = msg.twist.linear
        omega: Vector3 = msg.twist.angular
        xyzw = np.array([quat.x, quat.y, quat.z, quat.w])
        self.client.resetBasePositionAndOrientation(
            self.cube_id, [pos.x, pos.y, pos.z], xyzw
        )
        self.client.resetBaseVelocity(
            self.cube_id,
            [vel.x, vel.y, vel.z],
            [omega.x, omega.y, omega.z],
        )
        self.client.stepSimulation()


def main(args=None):
    rclpy.init(args=args)
    node = PybulletNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
