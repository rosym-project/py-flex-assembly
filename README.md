# py-flex-assembly
PyBullet-Based Simulation Environment for the Flexible Assembly Robotics Scenario

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