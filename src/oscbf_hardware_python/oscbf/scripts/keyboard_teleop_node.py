"""Improved class-based ROS2 teleop twist keyboard node


Based on https://github.com/ros2/teleop_twist_keyboard/blob/dashing/teleop_twist_keyboard.py
"""

import sys
import threading
import termios
import tty

import rclpy
from rclpy.node import Node
import geometry_msgs.msg


class TeleopTwistKeyboard(Node):
    def __init__(self):
        super().__init__("teleop_twist_keyboard")

        # Node parameters
        self.declare_parameter("stamped", False)
        self.declare_parameter("frame_id", "")

        self.stamped = self.get_parameter("stamped").value
        self.frame_id = self.get_parameter("frame_id").value

        # Validate parameters
        if not self.stamped and self.frame_id:
            raise Exception("'frame_id' can only be set when 'stamped' is True")

        # Choose message type based on stamped parameter
        self.twist_msg = (
            geometry_msgs.msg.TwistStamped if self.stamped else geometry_msgs.msg.Twist
        )

        # Create publisher
        self.pub = self.create_publisher(self.twist_msg, "cmd_vel", 10)

        # Movement and speed configurations
        self.move_bindings = {
            "w": (1, 0, 0, 0, 0, 0),  # forward (x+)
            "s": (-1, 0, 0, 0, 0, 0),  # backward (x-)
            "a": (0, 1, 0, 0, 0, 0),  # left (y+)
            "d": (0, -1, 0, 0, 0, 0),  # right (y-)
            "q": (0, 0, 1, 0, 0, 0),  # up (z+)
            "e": (0, 0, -1, 0, 0, 0),  # down (z-)
            "j": (0, 0, 0, 1, 0, 0),  # roll left
            "l": (0, 0, 0, -1, 0, 0),  # roll right
            "i": (0, 0, 0, 0, 1, 0),  # pitch up
            "k": (0, 0, 0, 0, -1, 0),  # pitch down
            "u": (0, 0, 0, 0, 0, 1),  # yaw left
            "o": (0, 0, 0, 0, 0, -1),  # yaw right
        }

        self.speed_bindings = {
            "z": (1.1, 1.1),
            "x": (0.9, 0.9),
        }

        # Initial speeds and movements
        self.lin_rate = 0.5
        self.ang_rate = 1.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Save terminal settings
        self.settings = self.save_terminal_settings()

        # Start keyboard input thread
        self.keyboard_thread = threading.Thread(target=self.run_keyboard_input)
        self.keyboard_thread.start()

    def save_terminal_settings(self):
        return termios.tcgetattr(sys.stdin)

    def restore_terminal_settings(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run_keyboard_input(self):
        twist_msg = self.twist_msg()

        if self.stamped:
            twist = twist_msg.twist
            twist_msg.header.frame_id = self.frame_id
        else:
            twist = twist_msg

        try:
            print("\nTeleop Twist Keyboard - Press keys to move robot")
            while rclpy.ok():
                key = self.get_key()

                # Handle movement and speed changes
                if key in self.move_bindings:
                    self.x, self.y, self.z, self.roll, self.pitch, self.yaw = (
                        self.move_bindings[key]
                    )
                elif key in self.speed_bindings:
                    self.lin_rate *= self.speed_bindings[key][0]
                    self.ang_rate *= self.speed_bindings[key][1]
                    self.get_logger().info(
                        f"Speed: {self.lin_rate}, Turn: {self.ang_rate}"
                    )
                elif key == "\x03":  # CTRL-C
                    break
                else:
                    # Stop movement if unrecognized key
                    self.x = self.y = self.z = 0.0
                    self.roll = self.pitch = self.yaw = 0.0

                # Update twist message
                if self.stamped:
                    twist_msg.header.stamp = self.get_clock().now().to_msg()

                twist.linear.x = self.x * self.lin_rate
                twist.linear.y = self.y * self.lin_rate
                twist.linear.z = self.z * self.lin_rate
                twist.angular.x = self.roll * self.ang_rate
                twist.angular.y = self.pitch * self.ang_rate
                twist.angular.z = self.yaw * self.ang_rate

                self.pub.publish(twist_msg)

        except Exception as e:
            self.get_logger().error(f"Error in keyboard input: {e}")
        finally:
            # Ensure robot stops when node is terminated
            twist.linear.x = twist.linear.y = twist.linear.z = 0.0
            twist.angular.x = twist.angular.y = twist.angular.z = 0.0
            self.pub.publish(twist_msg)
            self.restore_terminal_settings()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopTwistKeyboard()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
