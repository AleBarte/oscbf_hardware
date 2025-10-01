#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import time
from geometry_msgs.msg import Pose, Twist
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from oscbf_msgs.msg import EEState


class EEStateVisualizer(Node):
    def __init__(self):
        super().__init__("ee_state_visualizer")

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Create a subscriber to the ee_state topic
        self.subscription = self.create_subscription(
            EEState,  # Replace with actual message type if different
            "/ee_state",
            self.ee_state_callback,
            qos_profile,
        )

        # Data storage for plotting - using fixed-length arrays
        self.buffer_size = 100
        self.time_data = np.linspace(-self.buffer_size + 1, 0, self.buffer_size)

        # Position data (x, y, z)
        self.pos_x = np.zeros(self.buffer_size)
        self.pos_y = np.zeros(self.buffer_size)
        self.pos_z = np.zeros(self.buffer_size)

        # Quaternion data (x, y, z, w)
        self.quat_x = np.zeros(self.buffer_size)
        self.quat_y = np.zeros(self.buffer_size)
        self.quat_z = np.zeros(self.buffer_size)
        self.quat_w = np.zeros(self.buffer_size)

        # Linear velocity data (x, y, z)
        self.vel_x = np.zeros(self.buffer_size)
        self.vel_y = np.zeros(self.buffer_size)
        self.vel_z = np.zeros(self.buffer_size)

        # Angular velocity data (x, y, z)
        self.ang_vel_x = np.zeros(self.buffer_size)
        self.ang_vel_y = np.zeros(self.buffer_size)
        self.ang_vel_z = np.zeros(self.buffer_size)

        # Track data ranges for auto-scaling
        self.min_max_values = {
            "pos_x": [float("inf"), float("-inf")],
            "pos_y": [float("inf"), float("-inf")],
            "pos_z": [float("inf"), float("-inf")],
            "quat_x": [float("inf"), float("-inf")],
            "quat_y": [float("inf"), float("-inf")],
            "quat_z": [float("inf"), float("-inf")],
            "quat_w": [float("inf"), float("-inf")],
            "vel_x": [float("inf"), float("-inf")],
            "vel_y": [float("inf"), float("-inf")],
            "vel_z": [float("inf"), float("-inf")],
            "ang_vel_x": [float("inf"), float("-inf")],
            "ang_vel_y": [float("inf"), float("-inf")],
            "ang_vel_z": [float("inf"), float("-inf")],
        }

        # Data lock to synchronize access from callback and animation
        self.data_lock = threading.Lock()

        # Flag to indicate if we've received data
        self.data_received = False

        # Setup the plot
        self.setup_plot()

        self.get_logger().info("EE State Visualizer started")

    def update_min_max(self, name, value):
        """Update the min/max values for a data series"""
        if value < self.min_max_values[name][0]:
            self.min_max_values[name][0] = value
        if value > self.min_max_values[name][1]:
            self.min_max_values[name][1] = value

    def ee_state_callback(self, msg):
        try:
            with self.data_lock:
                # Update position data
                self.pos_x = np.roll(self.pos_x, -1)
                self.pos_y = np.roll(self.pos_y, -1)
                self.pos_z = np.roll(self.pos_z, -1)
                self.pos_x[-1] = msg.pose.position.x
                self.pos_y[-1] = msg.pose.position.y
                self.pos_z[-1] = msg.pose.position.z

                # Update min/max values for position
                self.update_min_max("pos_x", msg.pose.position.x)
                self.update_min_max("pos_y", msg.pose.position.y)
                self.update_min_max("pos_z", msg.pose.position.z)

                # Update quaternion data
                self.quat_x = np.roll(self.quat_x, -1)
                self.quat_y = np.roll(self.quat_y, -1)
                self.quat_z = np.roll(self.quat_z, -1)
                self.quat_w = np.roll(self.quat_w, -1)
                self.quat_x[-1] = msg.pose.orientation.x
                self.quat_y[-1] = msg.pose.orientation.y
                self.quat_z[-1] = msg.pose.orientation.z
                self.quat_w[-1] = msg.pose.orientation.w

                # Update min/max values for quaternion
                self.update_min_max("quat_x", msg.pose.orientation.x)
                self.update_min_max("quat_y", msg.pose.orientation.y)
                self.update_min_max("quat_z", msg.pose.orientation.z)
                self.update_min_max("quat_w", msg.pose.orientation.w)

                # Update linear velocity data
                self.vel_x = np.roll(self.vel_x, -1)
                self.vel_y = np.roll(self.vel_y, -1)
                self.vel_z = np.roll(self.vel_z, -1)
                self.vel_x[-1] = msg.twist.linear.x
                self.vel_y[-1] = msg.twist.linear.y
                self.vel_z[-1] = msg.twist.linear.z

                # Update min/max values for linear velocity
                self.update_min_max("vel_x", msg.twist.linear.x)
                self.update_min_max("vel_y", msg.twist.linear.y)
                self.update_min_max("vel_z", msg.twist.linear.z)

                # Update angular velocity data
                self.ang_vel_x = np.roll(self.ang_vel_x, -1)
                self.ang_vel_y = np.roll(self.ang_vel_y, -1)
                self.ang_vel_z = np.roll(self.ang_vel_z, -1)
                self.ang_vel_x[-1] = msg.twist.angular.x
                self.ang_vel_y[-1] = msg.twist.angular.y
                self.ang_vel_z[-1] = msg.twist.angular.z

                # Update min/max values for angular velocity
                self.update_min_max("ang_vel_x", msg.twist.angular.x)
                self.update_min_max("ang_vel_y", msg.twist.angular.y)
                self.update_min_max("ang_vel_z", msg.twist.angular.z)

                # Mark that we've received data
                self.data_received = True

            # Print some debug info occasionally (every ~20 callbacks)
            if np.random.random() < 0.05:
                self.get_logger().info(
                    f"Received EE State: Pos X={msg.pose.position.x:.4f}, "
                    f"Vel X={msg.twist.linear.x:.4f}"
                )
        except Exception as e:
            self.get_logger().error(f"Error in callback: {str(e)}")

    def setup_plot(self):
        # Create figure and subplots with more space between them
        self.fig, self.axs = plt.subplots(4, 4, figsize=(15, 10), dpi=100)
        plt.subplots_adjust(
            left=0.08, right=0.95, bottom=0.07, top=0.95, wspace=0.3, hspace=0.4
        )

        # Set fixed y-axis limits for plots
        quat_min, quat_max = -1.1, 1.1
        pos_min, pos_max = -1, 1
        vel_min, vel_max = -1, 1
        ang_vel_min, ang_vel_max = -1, 1

        # Set up position plots (row 0)
        (self.pos_x_line,) = self.axs[0, 0].plot(
            self.time_data, self.pos_x, "r-", linewidth=2
        )
        self.axs[0, 0].set_title("Position X", fontsize=12, fontweight="bold")
        self.axs[0, 0].set_xlabel("Time Steps", fontsize=10)
        self.axs[0, 0].set_ylabel("Position (m)", fontsize=10)
        self.axs[0, 0].set_ylim(pos_min, pos_max)
        self.axs[0, 0].grid(True)

        (self.pos_y_line,) = self.axs[0, 1].plot(
            self.time_data, self.pos_y, "g-", linewidth=2
        )
        self.axs[0, 1].set_title("Position Y", fontsize=12, fontweight="bold")
        self.axs[0, 1].set_xlabel("Time Steps", fontsize=10)
        self.axs[0, 1].set_ylim(pos_min, pos_max)
        self.axs[0, 1].grid(True)

        (self.pos_z_line,) = self.axs[0, 2].plot(
            self.time_data, self.pos_z, "b-", linewidth=2
        )
        self.axs[0, 2].set_title("Position Z", fontsize=12, fontweight="bold")
        self.axs[0, 2].set_xlabel("Time Steps", fontsize=10)
        self.axs[0, 2].set_ylim(pos_min, pos_max)
        self.axs[0, 2].grid(True)

        # Hide the unused plot in row 0
        self.axs[0, 3].set_visible(False)

        # Set up quaternion plots (row 1)
        (self.quat_x_line,) = self.axs[1, 0].plot(
            self.time_data, self.quat_x, "r-", linewidth=2
        )
        self.axs[1, 0].set_title("Quaternion X", fontsize=12, fontweight="bold")
        self.axs[1, 0].set_xlabel("Time Steps", fontsize=10)
        self.axs[1, 0].set_ylabel("Quaternion", fontsize=10)
        self.axs[1, 0].set_ylim(quat_min, quat_max)
        self.axs[1, 0].grid(True)

        (self.quat_y_line,) = self.axs[1, 1].plot(
            self.time_data, self.quat_y, "g-", linewidth=2
        )
        self.axs[1, 1].set_title("Quaternion Y", fontsize=12, fontweight="bold")
        self.axs[1, 1].set_xlabel("Time Steps", fontsize=10)
        self.axs[1, 1].set_ylim(quat_min, quat_max)
        self.axs[1, 1].grid(True)

        (self.quat_z_line,) = self.axs[1, 2].plot(
            self.time_data, self.quat_z, "b-", linewidth=2
        )
        self.axs[1, 2].set_title("Quaternion Z", fontsize=12, fontweight="bold")
        self.axs[1, 2].set_xlabel("Time Steps", fontsize=10)
        self.axs[1, 2].set_ylim(quat_min, quat_max)
        self.axs[1, 2].grid(True)

        (self.quat_w_line,) = self.axs[1, 3].plot(
            self.time_data, self.quat_w, "m-", linewidth=2
        )
        self.axs[1, 3].set_title("Quaternion W", fontsize=12, fontweight="bold")
        self.axs[1, 3].set_xlabel("Time Steps", fontsize=10)
        self.axs[1, 3].set_ylim(quat_min, quat_max)
        self.axs[1, 3].grid(True)

        # Set up linear velocity plots (row 2)
        (self.vel_x_line,) = self.axs[2, 0].plot(
            self.time_data, self.vel_x, "r-", linewidth=2
        )
        self.axs[2, 0].set_title("Linear Velocity X", fontsize=12, fontweight="bold")
        self.axs[2, 0].set_xlabel("Time Steps", fontsize=10)
        self.axs[2, 0].set_ylabel("Velocity (m/s)", fontsize=10)
        self.axs[2, 0].set_ylim(vel_min, vel_max)
        self.axs[2, 0].grid(True)

        (self.vel_y_line,) = self.axs[2, 1].plot(
            self.time_data, self.vel_y, "g-", linewidth=2
        )
        self.axs[2, 1].set_title("Linear Velocity Y", fontsize=12, fontweight="bold")
        self.axs[2, 1].set_xlabel("Time Steps", fontsize=10)
        self.axs[2, 1].set_ylim(vel_min, vel_max)
        self.axs[2, 1].grid(True)

        (self.vel_z_line,) = self.axs[2, 2].plot(
            self.time_data, self.vel_z, "b-", linewidth=2
        )
        self.axs[2, 2].set_title("Linear Velocity Z", fontsize=12, fontweight="bold")
        self.axs[2, 2].set_xlabel("Time Steps", fontsize=10)
        self.axs[2, 2].set_ylim(vel_min, vel_max)
        self.axs[2, 2].grid(True)

        # Hide the unused plot in row 2
        self.axs[2, 3].set_visible(False)

        # Set up angular velocity plots (row 3)
        (self.ang_vel_x_line,) = self.axs[3, 0].plot(
            self.time_data, self.ang_vel_x, "r-", linewidth=2
        )
        self.axs[3, 0].set_title("Angular Velocity X", fontsize=12, fontweight="bold")
        self.axs[3, 0].set_xlabel("Time Steps", fontsize=10)
        self.axs[3, 0].set_ylabel("Angular Velocity (rad/s)", fontsize=10)
        self.axs[3, 0].set_ylim(ang_vel_min, ang_vel_max)
        self.axs[3, 0].grid(True)

        (self.ang_vel_y_line,) = self.axs[3, 1].plot(
            self.time_data, self.ang_vel_y, "g-", linewidth=2
        )
        self.axs[3, 1].set_title("Angular Velocity Y", fontsize=12, fontweight="bold")
        self.axs[3, 1].set_xlabel("Time Steps", fontsize=10)
        self.axs[3, 1].set_ylim(ang_vel_min, ang_vel_max)
        self.axs[3, 1].grid(True)

        (self.ang_vel_z_line,) = self.axs[3, 2].plot(
            self.time_data, self.ang_vel_z, "b-", linewidth=2
        )
        self.axs[3, 2].set_title("Angular Velocity Z", fontsize=12, fontweight="bold")
        self.axs[3, 2].set_xlabel("Time Steps", fontsize=10)
        self.axs[3, 2].set_ylim(ang_vel_min, ang_vel_max)
        self.axs[3, 2].grid(True)

        # Hide the unused plot in row 3
        self.axs[3, 3].set_visible(False)

        # Set the figure title
        self.fig.suptitle(
            "End Effector State Visualization", fontsize=16, fontweight="bold"
        )

    def update_plot(self, frame):
        with self.data_lock:
            # Make local copies of the data to avoid race conditions
            pos_x = self.pos_x.copy()
            pos_y = self.pos_y.copy()
            pos_z = self.pos_z.copy()

            quat_x = self.quat_x.copy()
            quat_y = self.quat_y.copy()
            quat_z = self.quat_z.copy()
            quat_w = self.quat_w.copy()

            vel_x = self.vel_x.copy()
            vel_y = self.vel_y.copy()
            vel_z = self.vel_z.copy()

            ang_vel_x = self.ang_vel_x.copy()
            ang_vel_y = self.ang_vel_y.copy()
            ang_vel_z = self.ang_vel_z.copy()

        # Update position lines
        self.pos_x_line.set_ydata(pos_x)
        self.pos_y_line.set_ydata(pos_y)
        self.pos_z_line.set_ydata(pos_z)

        # Update quaternion lines
        self.quat_x_line.set_ydata(quat_x)
        self.quat_y_line.set_ydata(quat_y)
        self.quat_z_line.set_ydata(quat_z)
        self.quat_w_line.set_ydata(quat_w)

        # Update linear velocity lines
        self.vel_x_line.set_ydata(vel_x)
        self.vel_y_line.set_ydata(vel_y)
        self.vel_z_line.set_ydata(vel_z)

        # Update angular velocity lines
        self.ang_vel_x_line.set_ydata(ang_vel_x)
        self.ang_vel_y_line.set_ydata(ang_vel_y)
        self.ang_vel_z_line.set_ydata(ang_vel_z)

        return [
            self.pos_x_line,
            self.pos_y_line,
            self.pos_z_line,
            self.quat_x_line,
            self.quat_y_line,
            self.quat_z_line,
            self.quat_w_line,
            self.vel_x_line,
            self.vel_y_line,
            self.vel_z_line,
            self.ang_vel_x_line,
            self.ang_vel_y_line,
            self.ang_vel_z_line,
        ]

    def start_animation(self):
        # Create animation - explicitly set cache_frame_data to False
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=range(1000000),  # Use an explicit frame count instead of None
            interval=100,
            blit=True,
            cache_frame_data=False,
        )
        plt.show()


class ROS2Thread(threading.Thread):
    def __init__(self, node):
        threading.Thread.__init__(self)
        self.node = node
        self.daemon = True

    def run(self):
        rclpy.spin(self.node)


def main(args=None):
    rclpy.init(args=args)

    # Create the node
    visualizer = EEStateVisualizer()

    # Start ROS2 spin in separate thread
    ros_thread = ROS2Thread(visualizer)
    ros_thread.start()

    try:
        # Run matplotlib animation in the main thread
        visualizer.start_animation()
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
