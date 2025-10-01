# OSCBF + ROS2 + libfranka

A thin ROS2 wrapper around `libfranka` for OSCBF integration


## ROS2 Workspace Setup

### Clone repos

```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/danielpmorton/oscbf_libfranka_ros2
git clone https://github.com/danielpmorton/oscbf_franka_python oscbf
git clone https://github.com/danielpmorton/oscbf_msgs
git clone https://github.com/danielpmorton/libfranka_08_patch libfranka
```


```
cd ~/ros2_ws
colcon build
source install/setup.bash
```

```
cd ~/ros2_ws/src/oscbf
pyenv virtualenv 3.10.12 oscbf_3pt10
pyenv local oscbf_3pt10
pip install -e .
```


libfranka-sim
```
cd ~/software/libfranka-sim
# pyenv environment should activate automatically. if not, 
# pyenv shell libfrankasim
export MUJOCO_GL=glx
export PYOPENGL_PLATFORM=glx
run-franka-sim-server -v
```

Running with libfranka-sim
```
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
source ~/ros2_ws/install/setup.bash
ros2 run oscbf_libfranka_ros2 franka_impedance_controller --ros-args -p robot_hostname:=127.0.0.1 -p realtime:=false
```

Other terminals:

End-effector trajectory node
```
source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/oscbf
# pyenv environment should activate automatically. if not, 
# pyenv shell oscbf_3pt10
python oscbf/scripts/traj_node.py
```

OSCBF Franka control node
```
source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/oscbf
# pyenv environment should activate automatically. if not, 
# pyenv shell oscbf_3pt10
python oscbf/scripts/franka_control_node.py
```


### Install libfranka

First, make sure [my patched libfranka](https://github.com/danielpmorton/libfranka_08_patch) is cloned at `ros2_ws/src/libfranka`. Then, follow the instructions on the README of that repo. Note that there is no need to checkout the 0.8.0 tag anymore, and installing pinocchio is not needed.

### Additional steps

Might also need to run this:
```
cd ~/ros2_ws
rosdep install -i --from-path src --rosdistro humble -y
```

## Building

Some assorted notes:
```
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
colcon build --packages-select oscbf_libfranka_ros2 --cmake-args -DCMAKE_PREFIX_PATH=/opt/openrobots/lib/cmake
```
The cmake args might actually not be necessary now that I added the LD_LIBRARY_PATH arg?

## Install oculus

Follow the steps as found on the [oculus_reader github page](https://github.com/rail-berkeley/oculus_reader)
```
cd ~/software
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install # has to be run only once on a single user account
git clone https://github.com/rail-berkeley/oculus_reader
cd oculus_reader
git config lfs.https://github.com/rail-berkeley/oculus_reader.git/info/lfs.locksverify false
sudo apt install android-tools-adb
```
There are some additional steps on the Meta website to follow as well

Then, install the package in the virtual environment for the project:
```
cd ~/software/oculus_reader
pyenv shell oscbf_3pt10
pip install -e .
```

Notes:
- Go into settings and turn all of the automatic sleep times to the maximum value (4 hours)
- Add a sticker on top of the proximity sensor on the inside of the meta quest
- The Meta Quest 3 sometimes has some issues where it loses track of the controller, and then when it regains tracking, it "snaps" to the new location, leading to unstable robot control. The Quest 2 seems to be more stable here. 




## Notes from running the robot for hardware experiments, June 28


1. Connect to the Franka control box via ethernet
2. Make sure that the ethernet profile is configured to Franka (see the Setting Up the Network section of [Franka FCI documentation](https://frankarobotics.github.io/docs/getting_started.html) if this is not already configured)
3. Make sure that the robot joints are unlocked (accessed via the [Franka Desk](https://172.16.0.2/desk/)) and that the emergency stop button is not depressed. The robot should be in the blue light mode to begin accepting commands over FCI. This code also currently assumes that the Franka hand is attached.
4. Open 3 terminals and launch them in the following order:

**Terminal 1 (Franka node)**: Publishes joint states, subscribes to joint torques
```
cd ~/ros2_ws
source install/setup.bash
ros2 run oscbf_libfranka_ros2 franka_impedance_controller
```

**Terminal 2 (OSCBF node)**: Publishes joint torques, subscribes to joint states and desired EE state
```
cd ~/ros2_ws
source install/setup.bash
cd src/oscbf
# Pyenv virtual environment should activate automatically
python oscbf/scripts/franka_control_node.py
```

**Terminal 3 (Oculus node)**: Publishes desired EE state, subscribes to joint states
```
cd ~/ros2_ws
source install/setup.bash
cd src/oscbf
# Pyenv virtual environment should activate automatically
python oscbf/scripts/oculus_node.py
```

### Alternative terminal setup options

#### Trajectory following

Terminal 3 can be replaced with a trajectory node, which publishes just the desired EE state from a predefined EE trajectory
```
cd ~/ros2_ws
source install/setup.bash
cd src/oscbf
# Pyenv virtual environment should activate automatically
python oscbf/scripts/oculus_node.py
```

#### Testing in simulation

Terminal 1 can be replaced with a simulated pybullet environment which does not require a connection to the robot, and can be used to debug the controller prior to testing on hardware
```
cd ~/ros2_ws
source install/setup.bash
cd src/oscbf
# Pyenv virtual environment should activate automatically
python oscbf/scripts/pybullet_sim_node.py
```



### Debugging errors

- If the Franka node (Terminal 1) reports an error like `Move command rejected: command not possible in the current mode!`, depress and release the emergency stop to reset the mode. The robot should return to the blue light state and you can re-run the command
- 
