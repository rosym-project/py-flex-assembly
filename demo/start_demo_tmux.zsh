#!/bin/zsh

SESSION_NAME="flexassembly-demo"
BASE_DIR=${HOME}/system/flexassembly_dev_ws/

#TODO: replace commands
COMMAND_GRIPPER_SERVER="echo start gripper server"
COMMAND_VISION="echo start vision"
COMMAND_MOVEMENT_SERVER="echo start movement server"
COMMAND_DEMO_BUMP="echo bump"
COMMAND_DEMO_LEVER="echo lever"

############## START TMUX SESSION #####################################
attach() {
    if ! [ -n "${TMUX}" ]; then
        tmux attach-session -t ${SESSION_NAME}
    else
        tmux switch-client -t ${SESSION_NAME}
    fi
}

#test if session is already running and if yes connect/switch to it
tmux has-session -t ${SESSION_NAME}
if [ $? = "0" ]; then
    attach
    exit 0
fi

# create session
tmux new-session -d -s ${SESSION_NAME} -c ${BASE_DIR}

# start roscore in window 0
tmux new-window -d -t ${SESSION_NAME}:0 -n roscore -c ${BASE_DIR}
tmux rename-window -t ${SESSION_NAME}:0 "roscore"
tmux send-keys     -t ${SESSION_NAME}:0 "source /opt/ros/noetic/setup.zsh" Enter
tmux send-keys     -t ${SESSION_NAME}:0 "roscore" Enter

# create window for gripper server
tmux new-window -d -t ${SESSION_NAME}:1 -c ${BASE_DIR}
tmux rename-window -t ${SESSION_NAME}:1 "gripper"
tmux send-keys     -t ${SESSION_NAME}:1 "source devel/setup.zsh" Enter
tmux send-keys     -t ${SESSION_NAME}:1 "$COMMAND_GRIPPER_SERVER" Enter

# create window for movement server
tmux new-window -d -t ${SESSION_NAME}:2 -c ${BASE_DIR}
tmux rename-window -t ${SESSION_NAME}:2 "movement"
tmux send-keys     -t ${SESSION_NAME}:2 "source devel/setup.zsh" Enter
tmux send-keys     -t ${SESSION_NAME}:2 "source /opt/xbot/setup.zsh" Enter
tmux send-keys     -t ${SESSION_NAME}:2 "$COMMAND_MOVEMENT_SERVER" Enter

# create window for vision
tmux new-window -d -t ${SESSION_NAME}:3 -c ${BASE_DIR}
tmux rename-window -t ${SESSION_NAME}:3 "vision"
tmux send-keys     -t ${SESSION_NAME}:3 "source devel/setup.zsh" Enter
tmux send-keys     -t ${SESSION_NAME}:3 "$COMMAND_VISION" Enter

# create window for demo components
tmux new-window -d -t ${SESSION_NAME}:4 -c ${BASE_DIR}
tmux split-window  -t ${SESSION_NAME}:4 -v
tmux rename-window -t ${SESSION_NAME}:4 "demo"
tmux send-keys     -t ${SESSION_NAME}:4.0 "source devel/setup.zsh" Enter
tmux send-keys     -t ${SESSION_NAME}:4.0 "$COMMAND_DEMO_BUMP"
tmux send-keys     -t ${SESSION_NAME}:4.1 "cd $BASE_DIR" Enter
tmux send-keys     -t ${SESSION_NAME}:4.1 "source devel/setup.zsh" Enter
tmux send-keys     -t ${SESSION_NAME}:4.1 "$COMMAND_DEMO_LEVER"

# setup demo window as entry point
tmux select-window -t ${SESSION_NAME}:4

# attach to session
attach
