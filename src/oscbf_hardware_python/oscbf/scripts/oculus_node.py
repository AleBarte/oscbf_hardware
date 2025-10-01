"""Oculus ROS2 Node for teleoperating a robot (Franka Panda) with operational space control"""

from typing import Tuple
from functools import partial
import argparse

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from oculus_reader.reader import OculusReader

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from oscbf_msgs.msg import EEState
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import JointState

from oscbf.core.manipulator import load_panda, Manipulator, tuplify
from oscbf.utils.rotations_and_transforms import (
    invert_transform_numpy,
    Rx,
    Rz,
    construct_transform_numpy,
    rmat_to_quat,
    construct_transform,
    invert_transform,
    quaternion_angular_error_numpy,
    slerp_numpy,
)


class OculusNode(Node):
    """Oculus ROS2 Node for teleoperating a robot (Franka Panda) with operational space control

    Publishes: EEState (pose and twist info for the end-effector)
    Subscribes: JointState (joint positions and velocities of the robot)

    Args:
        publish_freq (float): Frequency at which to publish the end-effector state. Defaults to 1000 (Hz)
        oculus_freq (float): Frequency at which to read from the Oculus. Defaults to 50 (Hz)
        hand (str): Which controller to use ("left" or "right"). Defaults to "right"
        debug (bool): Whether to run in debugging mode (without needing to subscribe to joint states).
            Defaults to False
    """

    def __init__(
        self,
        publish_freq: float = 1000,
        oculus_freq: float = 50,
        hand: str = "right",
        debug: bool = False,
    ):
        super().__init__("oculus_node")

        assert publish_freq > 0, "Publishing frequency must be greater than 0"
        assert oculus_freq > 0, "Oculus frequency must be greater than 0"
        assert publish_freq >= oculus_freq
        if oculus_freq > 70:
            self.get_logger().warn(
                "Oculus frequency is set to a value greater than 70Hz. This may be too high for the Oculus"
            )
        assert hand in ["left", "right"], "Hand must be 'left' or 'right'"
        self.publsh_freq = float(publish_freq)
        self.oculus_freq = float(oculus_freq)
        self.hand = hand

        self.get_logger().info("Loading Franka model...")
        self.robot = load_panda()

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.oculus_reader = OculusReader()

        self.is_active = False

        self.last_joint_state = None
        self.last_oculus_read_time = 0.0

        if debug:
            # Set an initial joint state for debugging, without running the robot node
            q_init = np.array([0, -np.pi / 3, 0, -5 * np.pi / 6, 0, np.pi / 2, 0])
            self.last_joint_state = np.concatenate([q_init, np.zeros_like(q_init)])

        # Controller aligned to controller transformation (and the inverse).
        # Rotates the controller frame so that x is forward and z is up when the controller is held
        self.T_CA_to_C = construct_transform_numpy(
            Rx(np.deg2rad(-135)) @ Rz(np.deg2rad(90)), np.zeros(3)
        )
        self.T_C_to_CA = invert_transform_numpy(self.T_CA_to_C)
        self.T_CA_to_C_tup = tuplify(self.T_CA_to_C)  # Static tuple for JIT compilation

        # Headset to world transformation (and the inverse).
        self.T_H_to_W = None
        self.T_W_to_H = None
        self.T_H_to_W_tup = None  # Static tuple for JIT compilation

        # Headset to base transformation
        self.T_H_to_B = None

        # End-effector to controller transformation
        self.T_E_to_C = None

        # Exponential smoothing filter parameters. Larger values == less smoothing
        self.pose_filter_param = 0.05  # TODO check this value
        self.twist_filter_param = 0.01

        # Initialize stored values of the desired end-effector state (direct from oculus)
        # and the filtered end-effector state (for publishing smooth, high-frequency data)
        self.desired_position = None
        self.desired_quaternion = None
        self.desired_velocity = None
        self.desired_omega = None
        self.filtered_position = None
        self.filtered_quaternion = None
        self.filtered_velocity = None
        self.filtered_omega = None

        # Construct publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, "franka/joint_states", self.joint_state_callback, qos_profile
        )
        self.ee_state_pub = self.create_publisher(EEState, "ee_state", qos_profile)
        self.oculus_timer = self.create_timer(
            1 / self.oculus_freq, self.record_ee_state
        )
        self.pub_timer = self.create_timer(1 / self.publsh_freq, self.publish_ee_state)

    def record_ee_state(self) -> None:
        """Record the end-effector state from the Oculus"""

        if self.last_joint_state is None:
            self.get_logger().warn("No joint state received yet.")
            return

        # Read from oculus
        transforms, buttons = self.oculus_reader.get_transformations_and_buttons()

        if transforms == {}:
            self.get_logger().warn("No transformations received yet.")
            return

        # Record current time
        time_object = self.get_clock().now()
        secs, nanosecs = time_object.seconds_nanoseconds()
        current_time = secs + nanosecs / 1e9

        # Extract data from oculus dictionaries
        # Transformation matrix for the controller
        T_C_cur_to_H = transforms["r" if self.hand == "right" else "l"]
        # Trigger button state
        # trigger_pressed = buttons["RTr" if self.hand == "right" else "LTr"]
        # trigger_value = buttons["rightTrig" if self.hand == "right" else "leftTrig"][0]
        # Grip button state
        grip_pressed = buttons["RG" if self.hand == "right" else "LG"]
        # grip_value = buttons["rightGrip" if self.hand == "right" else "leftGrip"][0]
        # Joystick button state.
        # joystick_pressed = buttons["RJ" if self.hand == "right" else "LJ"]
        # joystick_values = buttons["rightJS" if self.hand == "right" else "leftJS"]
        # joystick_x_value = joystick_values[0]
        # joystick_y_value = joystick_values[1]
        # Other button states
        # a_or_x_pressed = buttons["A" if self.hand == "right" else "X"]
        # b_or_y_pressed = buttons["B" if self.hand == "right" else "Y"]

        # We want to be controlling the robot only if the grip button is pressed
        if grip_pressed:
            if not self.is_active:
                # Transition from inactive to active:
                self.is_active = True
                # If this is the first time we are active in this session, set the world transform
                if self.T_H_to_W is None or self.T_W_to_H is None:
                    self._set_world_origin(T_C_cur_to_H)
                # Store the transformation matrices we need
                self._cache_transforms(T_C_cur_to_H)

            # End-effector to base transformation
            desired_transform = self.T_H_to_B @ T_C_cur_to_H @ self.T_E_to_C
            q = rmat_to_quat(desired_transform[:3, :3])
            pos = desired_transform[:3, 3]

            if self.desired_position is None or self.desired_quaternion is None:
                # Edge case for the first command
                self.desired_velocity = np.zeros((3,))
                self.desired_omega = np.zeros((3,))
            else:
                dt = current_time - self.last_oculus_read_time
                self.desired_velocity = (pos - self.desired_position) / dt
                self.desired_omega = (
                    quaternion_angular_error_numpy(q, self.desired_quaternion) / dt
                )
            self.desired_position = pos
            self.desired_quaternion = q
            self.last_oculus_read_time = current_time

        else:
            if self.is_active:
                # Transition from active to inactive
                self.is_active = False
                # Stop at the last commanded pose
                self.desired_velocity = np.zeros((3,))
                self.desired_omega = np.zeros((3,))

    def publish_ee_state(self) -> None:
        """Publish a smoothed end-effector state, based on the last known transformations from the Oculus"""

        if self.desired_position is None:
            # No data to publish yet
            return

        # Edge cases for the first command. Note: If any are unset, they're all unset
        if self.filtered_position is None:
            self.filtered_position = self.desired_position
            self.filtered_quaternion = self.desired_quaternion
            self.filtered_velocity = self.desired_velocity
            self.filtered_omega = self.desired_omega

        # Apply exponential smoothing
        self.filtered_position = (
            self.pose_filter_param * self.desired_position
            + (1.0 - self.pose_filter_param) * self.filtered_position
        )
        self.filtered_quaternion = slerp_numpy(
            self.filtered_quaternion, self.desired_quaternion, self.pose_filter_param
        )
        self.filtered_velocity = (
            self.twist_filter_param * self.desired_velocity
            + (1.0 - self.twist_filter_param) * self.filtered_velocity
        )
        self.filtered_omega = (
            self.twist_filter_param * self.desired_omega
            + (1.0 - self.twist_filter_param) * self.filtered_omega
        )

        # Construct and publish the message
        msg = EEState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position = Point(
            x=self.filtered_position[0],
            y=self.filtered_position[1],
            z=self.filtered_position[2],
        )
        msg.pose.orientation = Quaternion(
            x=self.filtered_quaternion[0],
            y=self.filtered_quaternion[1],
            z=self.filtered_quaternion[2],
            w=self.filtered_quaternion[3],
        )
        msg.twist.linear = Vector3(
            x=self.filtered_velocity[0],
            y=self.filtered_velocity[1],
            z=self.filtered_velocity[2],
        )
        msg.twist.angular = Vector3(
            x=self.filtered_omega[0], y=self.filtered_omega[1], z=self.filtered_omega[2]
        )
        self.ee_state_pub.publish(msg)

    def _set_world_origin(self, T_C_cur_to_H: np.ndarray) -> None:
        """Set the world origin, assuming it is aligned with the current controller transform

        Args:
            T_C_cur_to_H (np.ndarray): Current controller-to-headset transformation, shape (4, 4)
        """
        T_CA_cur_to_H = T_C_cur_to_H @ self.T_CA_to_C
        self.T_W_to_H = T_CA_cur_to_H
        self.T_H_to_W = invert_transform_numpy(self.T_W_to_H)
        self.T_H_to_W_tup = tuplify(self.T_H_to_W)

    def _cache_transforms(self, T_C_cur_to_H: np.ndarray) -> None:
        """Cache the transforms that don't change during a single teleop trajectory

        These get updated every time the controller grip is released and re-pressed.

        Args:
            T_C_cur_to_H (np.ndarray): Current controller-to-headset transformation, shape (4, 4)
        """
        T_H_to_B, T_E_to_C = compute_cached_transforms(
            T_C_cur_to_H,
            self.last_joint_state[: self.robot.num_joints],
            self.robot,
            self.T_CA_to_C_tup,
            self.T_H_to_W_tup,
        )
        self.T_H_to_B = np.asarray(T_H_to_B)
        self.T_E_to_C = np.asarray(T_E_to_C)

    def joint_state_callback(self, msg: JointState):
        """Log the last known joint state of the robot (joint positions and velocities)

        Args:
            msg (JointState): Joint state message
        """
        self.last_joint_state = np.array([msg.position, msg.velocity]).ravel()


# Note: assuming that headset-to-world transformation is static... May change this eventually
@partial(jax.jit, static_argnums=(2, 3, 4))
def compute_cached_transforms(
    T_C_cur_to_H: Array, q: Array, robot: Manipulator, T_CA_to_C: Tuple, T_H_to_W: Tuple
) -> Tuple[Array, Array]:
    """Compute the transforms that we will cache whenever we start tracking the teleop motion

    Args:
        T_C_cur_to_H (Array): Current controller-to-handset transformation, shape (4, 4)
        q (Array): Latest joint angles of the robot, shape (num_joints,)
        robot (Manipulator): Robot model
        T_CA_to_C (Array): Controller-aligned-to-controller transformation, shape (4, 4)
        T_H_to_W (Array): Headset-to-world transformation, shape (4, 4)

    Returns:
        Tuple[Array, Array]:
            T_H_to_B: Headset-to-base transformation, shape (4, 4)
            T_E_to_C: End-effector-to-controller transformation, shape (4, 4)
    """
    # fmt: off
    T_CA_to_C = jnp.asarray(T_CA_to_C) # Controller aligned to controller
    T_H_to_W = jnp.asarray(T_H_to_W) # Headset to world
    # Compute inverses of inputs

    T_C_to_CA = invert_transform(T_CA_to_C) # Controller to controller aligned
    T_W_to_H = invert_transform(T_H_to_W) # World to headset
    # Compute end effector state
    T_E_ref_to_B = robot.ee_transform(q) # End-effector reference to base
    T_E_to_EA = construct_transform(T_E_ref_to_B[:3, :3], jnp.zeros(3)) # End-effector to end-effector aligned
    T_EA_to_E = invert_transform(T_E_to_EA) # End-effector aligned to end-effector
    # Compute controller state
    T_C_ref_to_H = T_C_cur_to_H # Controller reference to headset
    T_H_to_C_ref = invert_transform(T_C_ref_to_H) # Headset to controller reference
    T_W_to_C_ref = T_H_to_C_ref @ T_W_to_H # World to controller
    # Compute the main two transforms we need to store
    T_H_to_B = T_E_ref_to_B @ T_EA_to_E @ T_C_to_CA @ T_W_to_C_ref @ T_H_to_W # Headset to base
    T_E_to_C = T_CA_to_C @ T_E_to_EA # End-effector to controller
    # fmt: on
    return T_H_to_B, T_E_to_C


def main(args=None):
    parser = argparse.ArgumentParser(description="Run the Oculus Node.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugging mode (default: False)",
    )
    parsed_args = parser.parse_args(args)

    rclpy.init(args=args)
    node = OculusNode(debug=parsed_args.debug)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
