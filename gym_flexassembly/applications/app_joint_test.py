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

import pyquaternion

from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

from std_msgs.msg import Float32MultiArray

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Transform, Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectoryPoint
from cosima_world_state.srv import RequestTrajectory, RequestTrajectoryResponse

from sensor_msgs.msg import JointState

import threading

import numpy as np

class JointTest(object):
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

        self.joint_state_data = None
        self.joint_state_data_internal = None
        self.lock_joint_state = threading.Lock()
        rospy.Subscriber("kin/dyn/robot/state", JointState, self.listener_joint_state)
        
        self.cur_position = np.array([0,0,0,0,0,0,0])

        self.rate = rospy.Rate(500) # 10hz

        print("Waiting for Joint feedback from robot...")

        while self.joint_state_data_internal == None:
            self.lock_joint_state.acquire()
            self.joint_state_data_internal = self.joint_state_data
            self.lock_joint_state.release()
            self.rate.sleep()

        for i in range(7):
            self.cur_position[i] = self.joint_state_data_internal.position[i]

        print("Received Joint feedback from robot! " + str(self.cur_position))

        self.rate.sleep()
        time.sleep(1)

        target_position = np.array([1.5185,-0.27916,0.0,1.25519,0.0,0.0,0.0])

        # home = [1.5185,-0.27916,0.0,1.25519,0.0,-1.5709,0.0]

        out_position = self.cur_position

        jt_tmp = JointTrajectoryPoint(positions=list(target_position), velocities=[0]*7)
        self.pub_traj_joint.publish(jt_tmp)
        self.rate.sleep()

        # step = self.normalized(target_position - out_position, 0) * 0.0001
        # while math.fabs(np.linalg.norm(target_position - out_position)) > 0.0001:
        #     out_position = out_position + step
        #     jt_tmp = JointTrajectoryPoint(positions=list(out_position), velocities=[0]*7)
        #     self.pub_traj_joint.publish(jt_tmp)
        #     self.rate.sleep()

        # # HOMING
        

        self.rate.sleep()
        time.sleep(2)

        # print("Grasp")
        # rospy.wait_for_service('/gripper1/close_gripper')
        # try:
        #     close_g = rospy.ServiceProxy('/gripper1/close_gripper', Empty)
        #     close_g()
        # except rospy.ServiceException as e:
        #     print("Service call failed: %s"%e)
        
        # rate.sleep()
        # time.sleep(2)


        
        # while math.fabs(np.linalg.norm(target - cur)) > 0.0001:
        #     # print(math.fabs(np.linalg.norm(target - cur)))
        #     cur = cur + step
        #     # print(cur)
        #     ros_t.translation.x = cur[0]
        #     ros_t.translation.y = cur[1]
        #     ros_t.translation.z = cur[2]

        #     timeq = timeq + quadT
        #     if (timeq > 1.0):
        #         timeq = 1.0
        #     q = pyquaternion.Quaternion.slerp(cur_quat, target_quat, timeq)
        #     ros_t.rotation.x = q[1]
        #     ros_t.rotation.y = q[2]
        #     ros_t.rotation.z = q[3]
        #     ros_t.rotation.w = q[0]

        #     self.pub_traj.publish(cart_traj_point)
        #     rate.sleep()

        # ros_t.translation.x = target[0]
        # ros_t.translation.y = target[1]
        # ros_t.translation.z = target[2]
        # ros_t.rotation.x = target_quat[1]
        # ros_t.rotation.y = target_quat[2]
        # ros_t.rotation.z = target_quat[3]
        # ros_t.rotation.w = target_quat[0]
        # cur_quat = target_quat
        # self.pub_traj.publish(cart_traj_point)
        # rate.sleep()
        # time.sleep(3)

    def normalized(self,a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a))
        # l2[l2==0] = 1
        return a / l2

    def listener_joint_state(self, data):
        self.lock_joint_state.acquire()
        self.joint_state_data = data
        self.lock_joint_state.release()

if __name__ == "__main__":
    c = JointTest()