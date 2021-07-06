#!/bin/zsh

DIR_SCRIPT=$(dirname $0)
#TODO: setup correct gripper file
#GRIPPER_USB_DEVICE="/dev/ttyUSB0"
GRIPPER_USB_DEVICE="/dev/ttyUSB0"
#TODO: enter sunrise and fri ip addresses!
NETWORK_ADDRESSES=(172.31.1.10 192.170.10.2)

############## CHECK GRIPPER ########################################
echo -n "CHECK gripper device file $GRIPPER_USB_DEVICE..."
if ! [[ -e $GRIPPER_USB_DEVICE ]]; then
    echo "FAIL"
    echo "Gripper device file $GRIPPER_USB_DEVICE is not available!"
    echo "Maybe the USB cable is not connected?"
    exit 1
fi

if [[ -r $GRIPPER_USB_DEVICE && -w $GRIPPER_USB_DEVICE ]]; then
    echo "OKAY"
else
    echo "FAIL"
    echo "Gripper device file is not read/write accessible."
    echo "Please enter the sudo password to execute:"
    echo "sudo chmod o+rw $GRIPPER_USB_DEVICE"
    sudo chmod u+rw,g+rw,o+rw $GRIPPER_USB_DEVICE || exit 1
fi
echo

############## CHECK NETWORK #########################################
echo "CHECK network connections..."
for ADDRESS in $NETWORK_ADDRESSES
do
    echo -n "CHECK $ADDRESS..."
    ping -q -c 1 -W 1 $ADDRESS > /dev/null

    if [ $? -gt 0 ]
    then
        echo "FAIL"
        echo "$ADDRESS could not be reached!"
        echo "Verify that all cables are connected to the correct ports!"
        exit 1
    else
        echo "OKAY"
    fi
done
echo

############## CHECK CAMERA ##########################################
echo -n "CHECK camera..."
${DIR_SCRIPT}/verify_realsense.py
CAM_AVAILABLE=$?
if [ $CAM_AVAILABLE -eq 0 ]; then
    echo "OKAY"
else
    echo "FAIL"
    echo "Connection to the realsense camera could not be established."
    echo "Please connect the USB cable and start again."
    exit 1
fi

############## START TMUX SESSION #####################################
#attach() {
    #if ! [ -n "${TMUX}" ]; then
        #tmux attach-session -t ${SESSION_NAME}
    #else
        #tmux switch-client -t ${SESSION_NAME}
    #fi
#}

 ##test if session is already running and if yes connect/switch to it
#tmux has-session -t ${SESSION_NAME}
#if [ $? = "0" ]; then
    #attach
    #exit 0
#fi

## create session
#tmux new-session -d -s ${SESSION_NAME} -c ${BASE_DIR}

## start roscore in window 0
#tmux new-window -d -t ${SESSION_NAME}:0 -n roscore -c ${BASE_DIR}
#tmux rename-window -t ${SESSION_NAME}:0 "roscore"
#tmux send-keys     -t ${SESSION_NAME}:0 "source /opt/ros/noetic/setup.zsh" Enter
#tmux send-keys     -t ${SESSION_NAME}:0 "roscore" Enter

## create window for gripper server
#tmux new-window -d -t ${SESSION_NAME}:1 -c ${BASE_DIR}
#tmux rename-window -t ${SESSION_NAME}:1 "gripper"
#tmux send-keys     -t ${SESSION_NAME}:1 "source devel/setup.zsh" Enter
#tmux send-keys     -t ${SESSION_NAME}:1 "$COMMAND_GRIPPER_SERVER" Enter

## create window for movement server
#tmux new-window -d -t ${SESSION_NAME}:2 -c ${BASE_DIR}
#tmux rename-window -t ${SESSION_NAME}:2 "movement"
#tmux send-keys     -t ${SESSION_NAME}:2 "source devel/setup.zsh" Enter
#tmux send-keys     -t ${SESSION_NAME}:2 "source /opt/xbot/setup.zsh" Enter
#tmux send-keys     -t ${SESSION_NAME}:2 "$COMMAND_MOVEMENT_SERVER" Enter

## create window for vision
#tmux new-window -d -t ${SESSION_NAME}:3 -c ${BASE_DIR}
#tmux rename-window -t ${SESSION_NAME}:3 "vision"
#tmux send-keys     -t ${SESSION_NAME}:3 "source devel/setup.zsh" Enter
#tmux send-keys     -t ${SESSION_NAME}:3 "$COMMAND_VISION" Enter

## create window for gui control
#tmux new-window -d -t ${SESSION_NAME}:4 -c ${BASE_DIR}
#tmux rename-window -t ${SESSION_NAME}:4 "control_gui"
#tmux send-keys     -t ${SESSION_NAME}:4 "source devel/setup.zsh" Enter
#tmux send-keys     -t ${SESSION_NAME}:4 "$COMMAND_CONTROL_GUI" Enter

## setup control gui window as entry point
#tmux select-window -t ${SESSION_NAME}:4

## attach to session
#attach
