"""Grasping Testing for FlexAssembly.
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

class GraspTest(object):
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

        # # write trajectory command
        self.pub_traj = rospy.Publisher("/cart/traj_setpoint", MultiDOFJointTrajectoryPoint, queue_size=1,latch=True)

        # Create publisher for trajectory
        # pub_traj = rospy.Publisher('flex_planning_ros/traj_setpoints', JointTrajectoryPoint, queue_size=1,latch=True)
        # print(" > Initialized publisher on flex_planning_ros/traj_setpoints")

        # # HOMING
        # jt_tmp = JointTrajectoryPoint(positions=[1.5,-0.1,0.0,-2.0,0.0,1.0,1.57], velocities=[0]*7)
        # pub_traj.publish(jt_tmp)
        rate = rospy.Rate(10) # 10hz
        rate.sleep()
        rate.sleep()

        cart_traj_point = MultiDOFJointTrajectoryPoint()
        ros_t = Transform()
        ros_t.translation.x = 0.4
        ros_t.translation.y = 0.2
        ros_t.translation.z = 0.8
        ros_t.rotation.w = 0
        ros_t.rotation.x = 0
        ros_t.rotation.y = 1
        ros_t.rotation.z = 0
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

        time.sleep(4)

        cart_traj_point.transforms[0].translation.x = 0.4
        cart_traj_point.transforms[0].translation.y = 0.20000000000000007
        cart_traj_point.transforms[0].translation.z = 0.723

        self.pub_traj.publish(cart_traj_point)
        rate.sleep()

        time.sleep(4)

        return

        # Move to first target
        # time.sleep(10)
        print("Next state")
        rospy.wait_for_service('flex_planning_ros/plan')
        try:
            add_two_ints = rospy.ServiceProxy('flex_planning_ros/plan', RequestTrajectory)
            goal = Pose()
            goal.position.x = 0.4
            goal.position.y = 0.2
            goal.position.z = 0.8

            goal.orientation.x = 0
            goal.orientation.y = 1
            goal.orientation.z = 0
            goal.orientation.w = 0
            resp1 = add_two_ints(goal)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        
        rate.sleep()
        time.sleep(10)
        return

        print("Prepare Grasp")
        rospy.wait_for_service('flex_planning_ros/plan')
        try:
            add_two_ints = rospy.ServiceProxy('flex_planning_ros/plan', RequestTrajectory)
            goal = Pose()
            goal.position.x = 0.4
            goal.position.y = 0.2
            goal.position.z = 0.72

            goal.orientation.x = 0
            goal.orientation.y = 1
            goal.orientation.z = 0
            goal.orientation.w = 0
            resp1 = add_two_ints(goal)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        
        rate.sleep()
        time.sleep(10)
        print("Grasp")
        rospy.wait_for_service('/gripper1/close_gripper')
        try:
            close_g = rospy.ServiceProxy('/gripper1/close_gripper', Empty)
            close_g()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        
        rate.sleep()
        time.sleep(2)
        print("Go up")
        rospy.wait_for_service('flex_planning_ros/plan')
        try:
            add_two_ints = rospy.ServiceProxy('flex_planning_ros/plan', RequestTrajectory)
            goal = Pose()
            goal.position.x = 0.4
            goal.position.y = 0.2
            goal.position.z = 0.8

            goal.orientation.x = 0
            goal.orientation.y = 1
            goal.orientation.z = 0
            goal.orientation.w = 0
            resp1 = add_two_ints(goal)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(10)
        print("Go to rail")
        rospy.wait_for_service('flex_planning_ros/plan')
        try:
            add_two_ints = rospy.ServiceProxy('flex_planning_ros/plan', RequestTrajectory)
            goal = Pose()
            goal.position.x = 0.0
            goal.position.y = 0.4
            goal.position.z = 0.8

            goal.orientation.x = 0
            goal.orientation.y = 1
            goal.orientation.z = 0
            goal.orientation.w = 0
            resp1 = add_two_ints(goal)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(10)
        # print("Release")
        # rospy.wait_for_service('/gripper1/open_gripper')
        # try:
        #     open_g = rospy.ServiceProxy('/gripper1/open_gripper', Empty)
        #     open_g()
        # except rospy.ServiceException as e:
        #     print("Service call failed: %s"%e)
        
        # rate.sleep()
        # time.sleep(2)
        ####################################
        print("Rotate")
        rospy.wait_for_service('flex_planning_ros/plan')
        try:
            add_two_ints = rospy.ServiceProxy('flex_planning_ros/plan', RequestTrajectory)
            goal = Pose()
            goal.position.x = 0.0
            goal.position.y = 0.4
            goal.position.z = 0.8

            goal.orientation.x = 0
            goal.orientation.y = -0.924
            goal.orientation.z = -0.383
            goal.orientation.w = 0
            resp1 = add_two_ints(goal)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(10)
        print("Doooown")
        rospy.wait_for_service('flex_planning_ros/plan')
        try:
            add_two_ints = rospy.ServiceProxy('flex_planning_ros/plan', RequestTrajectory)
            goal = Pose()
            goal.position.x = 0.0
            goal.position.y = 0.4
            goal.position.z = 0.75

            goal.orientation.x = 0
            goal.orientation.y = -0.924
            goal.orientation.z = -0.383
            goal.orientation.w = 0
            resp1 = add_two_ints(goal)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(10)
        print("Finished")

if __name__ == "__main__":
    c = GraspTest()