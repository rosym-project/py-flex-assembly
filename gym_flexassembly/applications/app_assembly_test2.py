"""Assembly Testing for FlexAssembly.
:Author:
    `Dennis Leroy Wigand <dwigand@cor-lab.de>`
"""
import os

import time

import signal

import sys

import math

import rospy

from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

from std_msgs.msg import Float32MultiArray

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Transform, Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectoryPoint
from cosima_world_state.srv import RequestTrajectory, RequestTrajectoryResponse

import threading

import numpy as np

class AssemblyTest(object):
    def __init__(self):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('coord', anonymous=False)
        print("Init coord node")

        # receive current eef pose
        # rospy.Subscriber("/cur_ee_pose", Float32MultiArray, self.listener_cur_ee_pose)
        # rospy.Subscriber("/cart_imped_high/cartesian_pose", Pose, self.listener_cur_ee_pose)
        # self.var_cur_ee_pose = Pose()
        self.lock = threading.Lock()

        # Create publisher for trajectory
        self.pub_traj_joint = rospy.Publisher("flex_planning_ros/traj_setpoints", JointTrajectoryPoint, queue_size=1,latch=True)
        print(" > Initialized publisher on flex_planning_ros/traj_setpoints")

        # # write trajectory command
        self.pub_traj = rospy.Publisher("/cart/traj_setpoint", MultiDOFJointTrajectoryPoint, queue_size=1,latch=True)
        print(" > Initialized publisher on /cart/traj_setpoint")

        rate = rospy.Rate(500) # 10hz
        rate.sleep()
        rate.sleep()

        # # HOMING
        # jt_tmp = JointTrajectoryPoint(positions=[1.5,-0.1,0.0,-2.0,0.0,1.0,1.57], velocities=[0]*7)
        # self.pub_traj_joint.publish(jt_tmp)

        # rate.sleep()
        # time.sleep(3)

        cart_traj_point = MultiDOFJointTrajectoryPoint()
        ros_t = Transform()
        # ros_t.translation.x = 0.0
        # ros_t.translation.y = 0.42
        # ros_t.translation.z = 0.76
        # ros_t.rotation.w = 0.0
        # ros_t.rotation.x = 0.0
        # ros_t.rotation.y = -0.924
        # ros_t.rotation.z = -0.383

        ros_t.translation.x = 0.0
        ros_t.translation.y = 0.433
        ros_t.translation.z = 0.755
        ros_t.rotation.w = 0.0
        ros_t.rotation.x = 0.0
        ros_t.rotation.y = -0.924
        ros_t.rotation.z = -0.386
        cart_traj_point.transforms.append(ros_t)
        ros_tt = Twist()
        ros_tt.linear.x = 0
        ros_tt.linear.y = 0
        ros_tt.linear.z = 0
        ros_tt.angular.x = 0
        ros_tt.angular.y = 0
        ros_tt.angular.z = 0
        cart_traj_point.velocities.append(ros_tt)
        ros_ttt = Twist()
        ros_ttt.linear.x = 0
        ros_ttt.linear.y = 0
        ros_ttt.linear.z = 0
        ros_ttt.angular.x = 0
        ros_ttt.angular.y = 0
        ros_ttt.angular.z = 0
        cart_traj_point.accelerations.append(ros_ttt)
        self.pub_traj.publish(cart_traj_point)
        rate.sleep()

        # time.sleep(4)

        time.sleep(3)

    def normalized(self,a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a))
        # l2[l2==0] = 1
        return a / l2

if __name__ == "__main__":
    c = AssemblyTest()