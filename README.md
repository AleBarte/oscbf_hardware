# OSCBF ROS 2 Hardware

[![Paper](http://img.shields.io/badge/arXiv-2503.06736-B31B1B.svg)](https://arxiv.org/abs/2503.06736)

ROS 2 interfaces for fast and safe manipulator teleoperation with
[OSCBF](https://github.com/StanfordASL/oscbf).

Currently supported hardware platforms:

- Franka Emika Panda
- Universal Robots UR5e

## Repository structure

This repository contains two ROS 2 packages:

1. `oscbf_hardware_python`: OSCBF controllers, safety filters, simulation nodes,
   and command-source nodes.
2. `oscbf_msgs`: custom ROS 2 message definitions, including `EEState`.

The former `oscbf_hardware_cpp` package is no longer part of this repository.
Franka hardware communication and the hardware-facing torque controller are now
provided by `franka_ros2`.

The Python package is arranged as follows:

```text
oscbf_hardware_python/
├── oscbf_hardware_python/
│   ├── assets/
│   ├── scripts/
│   └── utils/
├── package.xml
└── setup.py
```

## Installation

### Python environment

The Python version used by the virtual environment must match the version used
by the ROS 2 installation. ROS 2 Humble normally uses Python 3.10, while ROS 2
Jazzy normally uses Python 3.12.

Install OSCBF in the same Python environment:

```bash
git clone https://github.com/StanfordASL/oscbf.git
cd oscbf
pip install -e .
```

See the [OSCBF repository](https://github.com/StanfordASL/oscbf) for its full
installation requirements.

### ROS 2 workspace

These instructions assume a workspace named `franka_ros2_ws`. Clone this
repository into its `src` directory:

```bash
mkdir -p ~/franka_ros2_ws/src
cd ~/franka_ros2_ws/src
git clone https://github.com/AleBarte/oscbf_hardware.git
```

Install the Python package in editable mode, then build and source the workspace:

```bash
cd ~/franka_ros2_ws/src/oscbf_hardware/oscbf_hardware_python
pip install -e .

cd ~/franka_ros2_ws
colcon build --symlink-install
source install/setup.bash
```

For Franka hardware, the workspace must also contain a compatible `franka_ros2`
checkout with the `forward_torque_controller` configured in `franka_bringup`.

## Franka Panda setup

The current Franka control path is:

```text
nominal torque source
    -> /joint_torque_controller/commands
    -> franka_forward_torque_node (OSCBF safety filter)
    -> /forward_torque_controller/commands
    -> franka_ros2 forward_torque_controller
```

Open a new terminal for each command and source the workspace first:

```bash
cd ~/franka_ros2_ws
source install/setup.bash
```

### Terminal 1: Franka hardware and torque controller

Replace `<ROBOT_IP>` with the IP address of the robot:

```bash
ros2 launch franka_bringup forward_torque_controller.launch.py \
  robot_ip:=<ROBOT_IP> arm_id:=fer load_gripper:=true use_rviz:=true
```

This publishes `/joint_states` and starts the controller that consumes filtered
torques from `/forward_torque_controller/commands`.

### Terminal 2: nominal torque source

Start the node or launch file that computes the nominal joint torques. It must
publish `std_msgs/msg/Float64MultiArray` messages on:

```text
/joint_torque_controller/commands
```

For example, when the `compliance_controller` package is installed in the same
workspace:

```bash
ros2 launch compliance_controller torque_control.launch.py use_oscbf:=true
```

### Terminal 3: OSCBF torque safety filter

```bash
ros2 run oscbf_hardware_python franka_forward_torque_node
```

The node waits until it has received both a joint state and a nominal torque
command before publishing a filtered command.

### Configuring OSCBF obstacles

Obstacles for the Franka safety filter are currently configured directly in
`oscbf_hardware_python/oscbf_hardware_python/scripts/franka_forward_torque.py`;
they are not ROS 2 parameters. Each obstacle is represented by a sphere with a
center in metres and a radius in metres. The configuration can be found at
[lines 151–166](oscbf_hardware_python/oscbf_hardware_python/scripts/franka_forward_torque.py#L151-L166):

```python
num_bodies = 2
all_collision_pos = np.array(
    [
        [0.0, 0.48, 0.15],
        [0.0, 0.48, 0.05],
    ]
)
all_collision_radii = np.repeat(0.05, len(all_collision_pos))

collision_pos = np.atleast_2d(all_collision_pos[:num_bodies])
collision_radii = all_collision_radii[:num_bodies]
```

Each row of `all_collision_pos` is an `[x, y, z]` sphere center. The values must
be expressed in the same base/world frame used by the OSCBF Panda model; the
node does not transform obstacle coordinates from TF. The corresponding entry
in `all_collision_radii` is that sphere's radius. `num_bodies` controls how many
entries from the beginning of both arrays are active.

The Panda links are also approximated by collision spheres. For every
robot-sphere/obstacle-sphere pair, the filter enforces:

```text
distance between centers - robot radius - obstacle radius >= 0
```

This calculation is implemented at
[lines 61–75](oscbf_hardware_python/oscbf_hardware_python/scripts/franka_forward_torque.py#L61-L75).

Consequently, increasing an obstacle radius adds a larger safety margin. Add
more rows to the position array (and matching radii) to approximate boxes,
cylinders, or other shapes with several spheres.

The same section of the file defines `z_min = 0.15`. Despite its name, this
value configures collision avoidance against a lateral plane, not a horizontal
table. The constraint is applied to coordinate index `1` (i.e. along the y
direction of the robot base frame) of the last seven robot collision spheres
and keeps each sphere, including its radius, on the allowed side of that plane.
In the current model frame, the plane is therefore located at `y = 0.15 m` with
respect to the robot base. The constraint is implemented at
[lines 89–94](oscbf_hardware_python/oscbf_hardware_python/scripts/franka_forward_torque.py#L89-L94).

After changing this configuration, restart `franka_forward_torque_node`. The
CBF is constructed and JIT-compiled only when the node starts. Self-collision
avoidance code is present at
[lines 77–87](oscbf_hardware_python/oscbf_hardware_python/scripts/franka_forward_torque.py#L77-L87),
but is currently disabled.

### Gazebo simulation

The hardware launch can be replaced by the Franka Gazebo launch:

```bash
ros2 launch franka_gazebo_bringup \
  gazebo_forward_torque_controller_example.launch.py \
  arm_id:=fer load_gripper:=true
```

Terminals 2 and 3 remain unchanged.

## Other installed executables

After building the workspace, the Python package also provides:

| Executable | Purpose |
| --- | --- |
| `controller` | End-effector pose-to-torque OSCBF controller |
| `ur5e_control_node` | UR5e OSCBF controller |
| `ee_traj_node` | Predefined desired end-effector trajectory publisher |
| `oculus_node` | Oculus desired end-effector state publisher |

Run an executable with:

```bash
ros2 run oscbf_hardware_python <EXECUTABLE>
```

The PyBullet simulation and debugging scripts are not registered as ROS 2
console executables. Run them from the package source tree, for example:

```bash
cd ~/franka_ros2_ws/src/oscbf_hardware/oscbf_hardware_python
python3 -m oscbf_hardware_python.scripts.pybullet_sim_node
```

## Hardware notes

- Connect the computer directly to the Franka control box over Ethernet and
  configure the network as described in the
  [Franka FCI documentation](https://frankarobotics.github.io/docs/getting_started.html).
- Unlock the robot joints in Franka Desk and release the emergency stop. The
  robot must be in the blue-light mode before it will accept FCI commands.
- The commands above assume that the Franka Hand is attached. Set
  `load_gripper:=false` if it is not.
- If the launch reports `Move command rejected: command not possible in the
  current mode!`, press and release the emergency stop, wait for the robot to
  return to the blue-light state, and launch again.
- When using a virtual environment, activate it before sourcing the workspace.
  Its Python version and ROS-related packages must be compatible with the ROS 2
  installation.

## Citation

```bibtex
@article{morton2025oscbf,
  author = {Morton, Daniel and Pavone, Marco},
  title = {Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions},
  year = {2025},
  journal = {arXiv preprint arXiv:2503.06736},
  note = {Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Hangzhou, 2025},
}
```
