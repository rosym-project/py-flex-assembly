# py-flex-assembly
PyBullet-Based Simulation Environment for the Flexible Assembly Robotics Scenario

## Installation of Software Stack

#### Prerequisites
* Ubuntu 20.04.1 LTS (Focal Fossa)
* ROS Noetic Ninjemys
* Bullet3 2.8.7 (commit hash: 93624761c606f020d2aae96d2086af4bb668e529)
* [PlotJuggler 2.8.3](https://github.com/facontidavide/PlotJuggler/releases/download/2.8.3/PlotJuggler-Linux-ROS-2.8.3.AppImage) (optional download)
* [RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense)

### Install ROS noetic

Setup your sources.list, keys, and reload
```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
```
Install ROS-Base: (Bare Bones) ROS packaging, build, and communication libraries. No GUI tools. 
```bash
sudo apt install ros-noetic-ros-base
```
Install ROS python dependencies for catkin (not catkin_make)
```bash
sudo apt install python3-pip python3-rosdep python3-catkin-tools python3-wstool python3-vcstool python3-osrf-pycommon python3-empy
```
Initialize ROS system dependency tool
```bash
sudo rosdep init
rosdep update
```

### Install RealSense SDK 2.0
* https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

### Install dependencies
#### Create a workspace for the dependencies
```bash
mkdir -p ~/system/flexassembly_deps_ws/src
cd ~/system/flexassembly_deps_ws
source /opt/ros/noetic/setup.bash
catkin config --init --install --extend /opt/ros/noetic/ --cmake-args -DCMAKE_BUILD_TYPE=Release
```
#### Build OROCOS RTT
```bash
cd ~/system/flexassembly_deps_ws/src
git clone -b toolchain-2.9 https://github.com/orocos-toolchain/rtt.git
git clone -b toolchain-2.9 https://github.com/orocos-toolchain/ocl.git
git clone -b toolchain-2.9 https://github.com/orocos/rtt_geometry.git
git clone -b toolchain-2.9 https://github.com/orocos/rtt_ros_integration.git
git clone -b toolchain-2.9 https://github.com/orocos-toolchain/log4cpp.git

cd ~/system/flexassembly_deps_ws
rosdep install --from-paths src --ignore-src --rosdistro noetic -y -r

sudo apt-get install liblua5.1-0-dev

catkin build
source devel/setup.bash
```
#### Build Bullet3
Install bullet in the same installation folder of the dependency workspace
```bash
cd ~/system/flexassembly_deps_ws/src
git clone https://github.com/bulletphysics/bullet3
cd bullet3
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/system/flexassembly_deps_ws/install -DCMAKE_BUILD_TYPE=Release -DUSE_DOUBLE_PRECISION=ON -DBUILD_SHARED_LIBS=ON -DBUILD_PYBULLET=OFF
make install
```

#### Install system dependencies
```bash
sudo apt install ros-noetic-tf ros-noetic-tf2 ros-noetic-tf2-eigen ros-noetic-tf2-geometry-msgs ros-noetic-tf2-kdl ros-noetic-tf2-msgs ros-noetic-tf2-ros ros-noetic-tf2-sensor-msgs ros-noetic-tf2-tools ros-noetic-tf-conversions ros-noetic-ros-base ros-noetic-kdl-conversions ros-noetic-kdl-parser  libyaml-cpp-dev liborocos-kdl-dev ros-trajectory-msgs ros-noetic-xacro
```

### Install code

#### Create a workspace for the code
```bash
mkdir -p ~/system/flexassembly_dev_ws/src
cd ~/system/flexassembly_dev_ws
source /opt/ros/noetic/setup.bash
catkin config --init --install --extend ~/system/flexassembly_deps_ws/install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

#### Build the python code
```bash
pip3 install pybullet-planning gym pyquaternion klampt
cd ~/system/flexassembly_dev_ws/src
git clone https://github.com/rosym-project/cosima_world_state.git
git clone https://github.com/rosym-project/py-flex-assembly.git
cd ~/system/flexassembly_dev_ws
rosdep install --from-paths src --ignore-src --rosdistro noetic -y -r
catkin build
source devel/setup.bash
```

#### Build the c++ code
```bash
cd ~/system/flexassembly_dev_ws/src
git clone -b noetic https://github.com/rosym-project/rtt-sim-embedded.git
git clone -b noetic https://github.com/rosym-project/cosima-controller.git
cd ~/system/flexassembly_dev_ws
rosdep install --from-paths src --ignore-src --rosdistro noetic -y -r
catkin build
source devel/setup.bash
```

## Usage

## Demo

Have a look at the [demo](./demo) description for how to setup and run a demonstration in the laboratory.

## General Part
```bash
source /opt/ros/noetic/setup.bash
roscore
```

### Python Part

Simulation component
```bash
cd ~/system/flexassembly_dev_ws/src/py-flex-assembly
source ../../devel/setup.bash
python3 -m gym_flexassembly.envs.flex_assembly_env extrigger
```

Motion planning component (needs to load the same environment as the simulation)
```bash
cd ~/system/flexassembly_dev_ws/src/py-flex-assembly
source ../../devel/setup.bash
python3 -m gym_flexassembly.planning.flex_planning_ros
```

Planning command test
```bash
cd ~/system/flexassembly_dev_ws/src/py-flex-assembly
source ../../devel/setup.bash
python3 -m gym_flexassembly.tests.ros_service_call_plan_path 0.1 -0.1 1.2
```

### OROCOS RTT Part
```bash
cd ~/system/flexassembly_dev_ws
source devel/setup.bash
rosrun rtt_ros deployer src/rtt-sim-embedded/scripts/test_ros_parameter_server.ops
```

#### Vision
Install additional python dependencies (best in a [virtual environment](https://docs.python.org/3/library/venv.html)):
```
cd ~/system/flexassembly_dev_ws/src/py-flex-assembly
pip install -r gym_flexassembly/vision/pose_detection/projection/requirements.txt
```
Optionally: Dowload a [side detection model](https://uni-bielefeld.sciebo.de/s/iYGVMwOUgvifyYE) to improve the pose estimation accuracy.

Usage:
* ``` python -m gym_flexassembly.vision.pose_detection.projection.pose_service --side_model <side_model> ```

The *pose_service* is a ros service which can be called to get the latest estimation of the pose of a clamp in robot coordinates.
It connects to an Intel RealSense camera and listens for the current arm position on the topic */robot/fdb/cart_pose_0*.
Per default the service runs on the topic */pose_estimation*.
This can be modified by providing the *--topic* argument.
In addition, the pose service can be used with a camera file recorded with the realsense-viewer by providing the *--file* argument.

### ROS Part
```bash
source /opt/ros/noetic/setup.bash
rosservice call /gripper1/close_gripper
```
## Modeling

Install MPS plaintext plugin: https://plugins.jetbrains.com/plugin/8444-com-dslfoundry-plaintextgen/

```bash
git clone -b 2019.3 https://code.cor-lab.de/git/orocos-dsl.kinematics-dsl.git
```
```bash
git clone https://github.com/rosym-project/compliant-interaction-dsl.git
```
```bash
git clone https://github.com/rosym-project/sot-qp-dsl.git
```
```bash
git clone -b 2019.3 https://github.com/rosym-project/dimensions-dsl.git
```

## Random stuff

```bash
source /opt/xbot/setup.sh
```

```bash
rostopic pub --once /traj geometry_msgs/Pose "{position: [-0.3,0.5,0.6], orientation: [1,0,0,0]}"
```

```bash
python3 -m gym_flexassembly.applications.app_table_wiping
```

# Motion Planning

Based on:
* https://pypi.org/project/pybullet-planning/
* https://github.com/yijiangh/pybullet_planning_tutorials

```
pip3 install pybullet-planning
pip3 install termcolor
```

Inverse Kinematics:
* https://www.youtube.com/watch?v=lZjsAiewqaE

# ATI Sensor

* Diameter: 77,0 mm
* Height: 36,0 mm
* Weight: 283,0 g
* All dimensions contain the adapterplate.

* 3D Model: https://www.ati-ia.com/products/ft/ft_models.aspx?id=Gamma

# Deployment and Launch

ROS Core
```bash
roscore

```

Gripper Server
```bash
source /home/kogrob/system/flexassembly_dev_ws/devel/setup.zsh

python3 /home/kogrob/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/applications/app_gripper_if.py

```

Control Architecture
```bash
source /home/kogrob/system/flexassembly_dev_ws/devel/setup.zsh
source /opt/xbot/setup.sh

rosrun rtt_ros deployer /home/kogrob/system/flexassembly_dev_ws/src/cosima-controller/scripts/real_tests/test_real_interface.ops

```

Coordination
```bash
source /home/kogrob/system/flexassembly_dev_ws/devel/setup.zsh

python3 /home/kogrob/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/applications/app_msgs.py

```
