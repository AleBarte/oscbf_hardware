# OSCBF ROS2 Workspace

Fast, safe manipulator teleoperation with [OSCBF](https://github.com/StanfordASL/oscbf)

Currently supported hardware platforms:
- Franka Emika Panda

## Installation

### Optional prerequisite: virtual environment

If using a virtual environment, keep in mind that while the OSCBF code runs on multiple python versions (tested: 3.10, 3.11, 3.12), to get it to work with ROS2, you'll need the python version to match with your ROS2 python version. For ROS2 Humble, this means 3.10.x, and for ROS2 Jazzy, this means 3.12.x. 

### Prerequisite: Libfranka

For newer robots, you can probably follow the standard setup details on the [libfranka Github](https://github.com/frankarobotics/libfranka). However, our lab hsa an older Panda, which requires libfranka 0.8.0. To get this to work, I had to make a minor change to libfranka which is available [here](https://github.com/danielpmorton/libfranka_08_patch).

### Prerequisite: Oculus Reader

Follow the steps as found on the [oculus_reader github page](https://github.com/rail-berkeley/oculus_reader). Then, install the package in your environment with
```
cd oculus_reader
pip install -e .
```

When working with the Quest hardware, the following tips might be useful:
- Go into settings and turn all of the automatic sleep times to the maximum value (4 hours)
- Add a sticker on top of the proximity sensor on the inside of the headset
- The Meta Quest 3 sometimes has some issues where it loses track of the controller, and then when it regains tracking, it "snaps" to the new location, leading to unstable robot control. The Quest 2 seems to be more stable here.

### OSCBF

Run the following to download the OSCBF code and pip install it in your python environment. Note: you can clone this to whatever directory you prefer. 
```
git clone https://github.com/stanfordasl/oscbf
cd oscbf
pip install -e .
```
See the README on the [OSCBF Github](https://github.com/StanfordASL/oscbf) for additional details

### OSCBF ROS2

Finally, you'll need to clone this repo for the ROS2 hardware code
```
git clone https://github.com/StanfordASL/oscbf_hardware_ws
```
To build, run the following (Remember to `source /opt/ros/YOUR_ROS2_VERSION/setup.bash` beforehand)
```
cd oscbf_hardware_ws
colcon build
source install/setup.bash
```

## Overview

This contains three packages:

1. `oscbf_hardware_cpp`: C++ ROS2/libfranka interface
2. `oscbf_hardware_python`: Python ROS2/OSCBF interface
3. `oscbf_msgs`: Custom ROS2 message definitions

## Setup

**Terminal 1 (Franka node)**: Publishes joint states, subscribes to joint torques
```
cd oscbf_hardware_ws
source install/setup.bash
ros2 run oscbf_hardware_cpp franka_impedance_controller
```

**Terminal 2 (OSCBF node)**: Publishes joint torques, subscribes to joint states and desired EE state
```
cd oscbf_hardware_ws
source install/setup.bash
cd src/oscbf_hardware_python
python oscbf_hardware_python/scripts/franka_control_node.py
```

**Terminal 3 (Oculus node)**: Publishes desired EE state, subscribes to joint states
```
cd oscbf_hardware_ws
source install/setup.bash
cd src/oscbf_hardware_python
python oscbf_hardware_python/scripts/oculus_node.py
```

### Alternative terminal setup options

#### Trajectory following

Terminal 3 can be replaced with a trajectory node, which publishes just the desired EE state from a predefined EE trajectory
```
cd oscbf_hardware_ws
source install/setup.bash
cd src/oscbf_hardware_python
python oscbf_hardware_python/scripts/oculus_node.py
```

#### Testing in simulation

Terminal 1 can be replaced with a simulated pybullet environment which does not require a connection to the robot, and can be used to debug the controller prior to testing on hardware
```
cd oscbf_hardware_ws
source install/setup.bash
cd src/oscbf_hardware_python
python oscbf_hardware_python/scripts/pybullet_sim_node.py
```

## Assorted notes

- Connect to the Franka control box via ethernet
- Make sure that the ethernet profile is configured to Franka (see the Setting Up the Network section of [Franka FCI documentation](https://frankarobotics.github.io/docs/getting_started.html) if this is not already configured)
- Make sure that the robot joints are unlocked (accessed via the [Franka Desk](https://172.16.0.2/desk/)) and that the emergency stop button is not depressed. The robot should be in the blue light mode to begin accepting commands over FCI. This code also currently assumes that the Franka hand is attached.
- If the Franka node (Terminal 1) reports an error like `Move command rejected: command not possible in the current mode!`, depress and release the emergency stop to reset the mode. The robot should return to the blue light state and you can re-run the command


## Citation
```
@article{morton2025oscbf,
      author = {Morton, Daniel and Pavone, Marco},
      title = {Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions},
      year = {2025},
      journal = {arXiv preprint arXiv:2503.06736},
      note = {Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Hangzhou, 2025},
      }
```
