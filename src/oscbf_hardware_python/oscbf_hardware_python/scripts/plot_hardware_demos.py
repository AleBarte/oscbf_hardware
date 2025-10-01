"""Plotting rosbag data from the periodic unsafe motion experiment"""

import os
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from oscbf_hardware_python.core.manipulator import load_panda
from read_rosbags import load_ros2_bag


# plt.style.use("tableau-colorblind10")
# plt.style.use("fivethirtyeight")
# plt.style.use("ggplot")
# plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("seaborn-v0_8-white")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True
# NOTE: if there is an error with findfont here,
# sudo apt install msttcorefonts -qq
# rm ~/.cache/matplotlib -rf
# Also, to install latex, sudo apt-get install texlive texstudio texlive-latex-extra cm-super dvipng

BAG_DIR = "/home/dmorton/ros2_ws/bag_files"

ROBOT = load_panda()


def list_folders_in_directory(path: str) -> List[str]:
    """Lists all immediate subdirectories within a given path.

    Args:
        path (str): The path to the directory to scan.

    Returns:
        list: A list of the full paths of all subdirectories.
    """
    folders = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            folders.append(full_path)
    return folders


# fmt: off
BAGS = [
    "/home/dmorton/ros2_ws/bag_files/rosbag2_2025_06_28-11_46_33", # Whole body workspace, angular freq = 2, max whole body x position = 0.75
    "/home/dmorton/ros2_ws/bag_files/rosbag2_2025_06_28-12_14_14", # EE workspace, angular freq = 3. Possibly wrong joint vel cbf
    "/home/dmorton/ros2_ws/bag_files/rosbag2_2025_06_28-12_24_37", # EE workspace, angular freq = 4. Possibly wrong joint vel cbf
    "/home/dmorton/ros2_ws/bag_files/rosbag2_2025_06_28-12_29_10", # EE workspace, angular freq = 5. Possibly wrong joint vel cbf
    "/home/dmorton/ros2_ws/bag_files/rosbag2_2025_06_28-14_49_10", # EE workspace, angular freq = 3. FIXED joint vel cbf
    "/home/dmorton/ros2_ws/bag_files/rosbag2_2025_06_28-15_55_46", # Singularity avoidance
]
# fmt: on


@jax.jit
@jax.vmap
def robot_ee_xpos(q):
    return ROBOT.ee_position(q)[0]


@jax.jit
@jax.vmap
def robot_whole_body_max_xpos(q):
    collision_pos_rad = ROBOT.link_collision_data(q)
    collision_pos = collision_pos_rad[:, :3]
    collision_radii = collision_pos_rad[:, 3]
    max_xpos = collision_pos[:, 0] + collision_radii
    return jnp.max(max_xpos)


def plot_whole_body_workspace_data() -> None:

    path = "/home/dmorton/ros2_ws/bag_files/rosbag2_2025_06_28-11_46_33"
    bag_dict = load_ros2_bag(path)

    ee_state_msgs = bag_dict["/ee_state"]
    joint_state_msgs = bag_dict["/franka/joint_states"]
    torque_cmd_msgs = bag_dict["/franka/torque_command"]

    desired_ee_xs = []
    ee_state_timestamps = []
    for msg in ee_state_msgs:
        desired_ee_xs.append(msg["pose"]["position"]["x"])
        ee_state_timestamps.append(msg["timestamp"] / 1e9)  # Convert ns to s

    joint_positions = []
    joint_velocities = []
    joint_state_timestamps = []
    for msg in joint_state_msgs:
        joint_positions.append(msg["position"])
        joint_velocities.append(msg["velocity"])
        joint_state_timestamps.append(msg["timestamp"] / 1e9)  # Convert ns to s

    torques = []
    torque_timesteps = []
    for msg in torque_cmd_msgs:
        torques.append(msg["data"])
        torque_timesteps.append(msg["timestamp"] / 1e9)  # Convert ns to s

    # Convert to numpy
    desired_ee_xs = np.asarray(desired_ee_xs)
    ee_state_timestamps = np.asarray(ee_state_timestamps)
    joint_positions = np.asarray(joint_positions)
    joint_velocities = np.asarray(joint_positions)
    joint_state_timestamps = np.asarray(joint_state_timestamps)
    torques = np.asarray(torques)
    torque_timesteps = np.asarray(torque_timesteps)

    # Align timestamps
    start_time = min(
        ee_state_timestamps[0], joint_state_timestamps[0], torque_timesteps[0]
    )
    ee_state_timestamps -= start_time
    joint_state_timestamps -= start_time
    torque_timesteps -= start_time

    # true_ee_xs = robot_ee_xpos(joint_positions)
    whole_body_max_xpos = robot_whole_body_max_xpos(joint_positions)

    # NOTE: the CBFs for this experiment operated on the whole body of the robot whereas the
    # desired reference command is only specified at the robot tip. So, if we want to determine
    # what the collision distance would be if we followed the unsafe reference, we need to
    # calibrate based on the robot configuration at the boundary of safety
    # eval_timestep = 13600
    # eval_joint_pos = np.atleast_2d(joint_positions[eval_timestep])
    # eval_ee_pos = robot_ee_xpos(eval_joint_pos).squeeze()
    # eval_wb_pos = robot_whole_body_max_xpos(eval_joint_pos).squeeze()
    # wb_to_ee_delta = eval_wb_pos - eval_ee_pos

    wb_to_ee_delta = 0.08493137589052924  # Calibrated value from above code
    whole_body_x_limit = 0.75  # Value from experiment
    h_unsafe = whole_body_x_limit - desired_ee_xs - wb_to_ee_delta
    h_safe = whole_body_x_limit - whole_body_max_xpos

    # We don't need all of the data, so restrict the plot limits to just a period or two
    t_start = 11.08
    t_end = 17.36
    safe_idx_start = np.searchsorted(joint_state_timestamps, t_start)
    safe_idx_end = np.searchsorted(joint_state_timestamps, t_end)
    unsafe_idx_start = np.searchsorted(ee_state_timestamps, t_start)
    unsafe_idx_end = np.searchsorted(ee_state_timestamps, t_end)

    unsafe_data_x = (
        ee_state_timestamps[unsafe_idx_start:unsafe_idx_end]
        - ee_state_timestamps[unsafe_idx_start]
    )
    unsafe_data_y = h_unsafe[unsafe_idx_start:unsafe_idx_end]
    safe_data_x = (
        joint_state_timestamps[safe_idx_start:safe_idx_end]
        - joint_state_timestamps[safe_idx_start]
    )
    safe_data_y = h_safe[safe_idx_start:safe_idx_end]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(unsafe_data_x, unsafe_data_y, label="Unsafe reference")
    ax.plot(safe_data_x, safe_data_y, label="Safe response")
    ax.plot([0, unsafe_data_x[-1]], [0, 0], "k--")

    ax.set_xlim([0, max(unsafe_data_x[-1], safe_data_x[-1])])

    # Add red region under y = 0
    ymin = ax.get_ylim()[0]
    new_ymin = 1 * ymin
    ax.axhspan(new_ymin, 0, color="red", alpha=0.1)
    ax.set_ylim(new_ymin, ax.get_ylim()[1])

    # ax.set_title("Whole-body workspace containment")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$h(\mathbf{z})$")  # : Distance to workspace violation")
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=14, loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_whole_body_workspace_data()
