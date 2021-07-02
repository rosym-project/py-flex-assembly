# Demo
This document describes how to setup and run a demonstration of the current state in our laboratory.

## Setup
The setup is divided into the setup steps for the robot and the setup steps on the compute.
The order should be:
1. Robot
2. Computer

### Robot
1. Switch on the robot
2. (Optional) Charge the robot
  1. Connect the charging cable
  2. Select the KMR robot on the controller
  3. Activate the charging process by navigating to BMS, then the socket charge menu and then pressing the charing button
3. Select the IIWA robot
4. Put the robot into its automatic control mode (turn the key, select AUT and then turn the key back)
5. Select the T1FRITorqueControl Application and start it

### Computer
1. Turn on the computer (_Note:_ Sometimes it does not boot if USB devices are connected. Then just unplug all USB cables.)
2. Connect all required cables:
  1. Two LAN cables have to be connected to the correct ports on the robot as well as the correct ports of the computer.
  2. Connect the USB cable of the camera. For the best camera video resolution, it has to be connected to the port marked with SS on the back of the computer.
  3. Connect the USB cable of the gripper.
3. Run [verify_demo.zsh](./verify_demo.zsh) to verify that all cables are connected correctly. The script will also query you to provide the sudo password to setup permissions for the gripper device file if not already set.
4. Now there are two options: If you are familiar with tmux, you can start the script [start_demo_tmux.zsh](./start_demo_tmux.zsh) which will start all required components in a tmux session, or you start them by hand:
  1. Terminal 1:
    1. `source /opt/ros/noetic/setup.zsh`
    2. `roscore`
  2. Terminal 2:
    1. `sudo chmod 777 /dev/ttyUSB0`
    2. `TODO gripper command`
  3. Terminal 3: (_Note:_ the order in which the envs are sourced is important)
    1. `source ${HOME}/system/flexassembly_dev_ws/devel/setup.zsh`
    2. `source /opt/xbot/setup.sh`
    3. `TODO server command`
  4. Terminal 4:
    1. `source ${HOME}/system/flexassembly_dev_ws/devel/setup.zsh`
    2. `TODO vision command`
  5. Terminal 5:
    1. `source ${HOME}/system/flexassembly_dev_ws/devel/setup.zsh`
5. In terminal 5 (tmux window demo), the commands for starting the demo have to be executed. In the tmux session they will already be written out so that only pressing enter remains. (_ATTENTION:_ currently the vision only supports the detection of a single clamp. Thus only a single one should be placed on the table. In addition, wile the scripts are running make sure to have a hand on the emergency shutdown at all times.) There are two options:
  1. `TODO command` This can only be used to place the first clamp, because of the way the robot moves the clamp on the rail.
  2. `TODO command` This command can be used either to place the first or the second clamp on the rail.

## Known Failures
* Sometimes the movement component will crash. Then, a red error message is displayed on the robot controller. To fix this, perform the following steps:
  1. Stop the movement server on the computer (started in Terminal 3, tmux window movement).
  2. Restart the application on the robot (step 5.).
  3. Restart the movement server on the computer.

