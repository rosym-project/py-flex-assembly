#!/home/dwigand/code/cogimon/CoSimA/pyBullet/vPyBullet/bin/python3
import os

import time

import signal

import sys

import rospy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cosima_world_state.srv import RequestTrajectory, RequestTrajectoryResponse

import numpy as np

rospy.init_node('sendjointangles', anonymous=False)
print("Init send joint angles node")

# Create publisher for trajectory
pub_traj = rospy.Publisher('flex_planning_ros/traj_setpoints', JointTrajectoryPoint, queue_size=1,latch=True)
print(" > Initialized publisher on flex_planning_ros/traj_setpoints")
############ TEST ONE SHOT!
jt_tmp = JointTrajectoryPoint(positions=[float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7])], velocities=[0]*7)
pub_traj.publish(jt_tmp)
print(jt_tmp)
#########

rate = rospy.Rate(10) # 10hz
rate.sleep()

rate.sleep()

rate.sleep()

# /home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/tests/ros_joint_pub.py