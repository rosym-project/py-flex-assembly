"""Polishing Testing for FlexAssembly.
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

from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Transform, Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectoryPoint
from cosima_world_state.srv import RequestTrajectory, RequestTrajectoryResponse

import pyquaternion

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

        # receive self.current eef pose
        rospy.Subscriber("/env/ft/ft_0", Wrench, self.listener_ft_wrench)
        self.wrench_data = None
        self.wrench_data_internal = None
        
        rospy.Subscriber("/robot/fdb/cart_pose_0", Pose, self.listener_cart_pose)
        self.pose_data = None
        self.pose_data_internal = None

        self.lock_wrench = threading.Lock()
        self.lock_pose = threading.Lock()

        # write trajectory command
        self.pub_traj = rospy.Publisher("/cart/traj_setpoint", MultiDOFJointTrajectoryPoint, queue_size=1,latch=True)

        # Wait for the first pose feedback
        self.rate = rospy.Rate(500) # 10hz

        print("Waiting for Pose feedback from robot...")

        while self.pose_data_internal == None:
            self.lock_pose.acquire()
            self.pose_data_internal = self.pose_data
            self.lock_pose.release()
            self.rate.sleep()

        print("Received Pose feedback from robot!")

        self.cart_traj_point = MultiDOFJointTrajectoryPoint()
        self.ros_t = Transform()
        self.ros_t.translation.x = self.pose_data_internal.position.x
        self.ros_t.translation.y = self.pose_data_internal.position.y
        self.ros_t.translation.z = self.pose_data_internal.position.z
        self.cur = np.array([self.ros_t.translation.x, self.ros_t.translation.y, self.ros_t.translation.z])

        self.ros_t.rotation.w = self.pose_data_internal.orientation.w
        self.ros_t.rotation.x = self.pose_data_internal.orientation.x
        self.ros_t.rotation.y = self.pose_data_internal.orientation.y
        self.ros_t.rotation.z = self.pose_data_internal.orientation.z
        self.cur_quat = pyquaternion.Quaternion(w=self.ros_t.rotation.w,x=self.ros_t.rotation.x,y=self.ros_t.rotation.y,z=self.ros_t.rotation.z)

        self.cart_traj_point.transforms.append(self.ros_t)
        ros_tt = Twist()
        ros_tt.linear.x = 0
        ros_tt.linear.y = 0
        ros_tt.linear.z = 0
        ros_tt.angular.x = 0
        ros_tt.angular.y = 0
        ros_tt.angular.z = 0
        self.cart_traj_point.velocities.append(ros_tt)
        ros_ttt = Twist()
        ros_ttt.linear.x = 0
        ros_ttt.linear.y = 0
        ros_ttt.linear.z = 0
        ros_ttt.angular.x = 0
        ros_ttt.angular.y = 0
        ros_ttt.angular.z = 0
        self.cart_traj_point.accelerations.append(ros_ttt)

        self.pub_traj.publish(self.cart_traj_point)

        print("Feed same position back!")

        self.rate.sleep()


        time.sleep(1)
        
        speed = 0.0001
        solo_tor_speed = 0.0005

        target_quat_left = pyquaternion.Quaternion(w=0.707,x=0,y=0.707,z=0) * pyquaternion.Quaternion(axis=[0, 1, 0], angle=25.0 / 180.0 * 3.14159265)
        # target_quat_left = pyquaternion.Quaternion(w=0.707,x=0,y=0.707,z=0)

        self.moveTo(target=np.array([0.8,0.0,0.75]), target_quat=target_quat_left, step_width=speed, solo_rot_step=solo_tor_speed)
        time.sleep(2)

        # self.moveTo(target=np.array([0.6,0.0,0.7]), target_quat=(target_quat_left * pyquaternion.Quaternion(axis=[0, 1, 0], angle=25.0 / 180.0 * 3.14159265)), step_width=speed, solo_rot_step=solo_tor_speed)

        # self.moveTo(target=np.array([0.8,0.0,0.5]), target_quat=(target_quat_left * pyquaternion.Quaternion(axis=[0, 1, 0], angle=25.0 / 180.0 * 3.14159265)), step_width=speed, solo_rot_step=solo_tor_speed)

        self.moveTo(target=np.array([0.8,0.0,0.5]), target_quat=target_quat_left, step_width=speed, solo_rot_step=solo_tor_speed)
        
        print("Finished")

    def listener_ft_wrench(self, data):
        self.lock_wrench.acquire()
        self.wrench_data = data
        self.lock_wrench.release()

    def listener_cart_pose(self, data):
        self.lock_pose.acquire()
        self.pose_data = data
        self.lock_pose.release()

    def moveTo(self, target, target_quat, step_width=0.0001, solo_rot_step=0.01):
        # Quaternion(w, x, y, z)
        leng = math.fabs(np.linalg.norm(target - self.cur))
        quadT = solo_rot_step
        timeq = 0
        if leng > 0:
            # quadT = 1.0/(leng) / step_width
        
            step = self.normalized(target - self.cur,0) * step_width
            while math.fabs(np.linalg.norm(target - self.cur)) > step_width:
                # print(math.fabs(np.linalg.norm(target - self.cur)))
                self.cur = self.cur + step
                # print(self.cur)
                self.ros_t.translation.x = self.cur[0]
                self.ros_t.translation.y = self.cur[1]
                self.ros_t.translation.z = self.cur[2]

                timeq = timeq + quadT
                if (timeq > 1.0):
                    timeq = 1.0
                q = pyquaternion.Quaternion.slerp(self.cur_quat, target_quat, timeq)
                self.ros_t.rotation.x = q[1]
                self.ros_t.rotation.y = q[2]
                self.ros_t.rotation.z = q[3]
                self.ros_t.rotation.w = q[0]

                self.pub_traj.publish(self.cart_traj_point)
                self.rate.sleep()
        else:
            quadT = solo_rot_step

        self.ros_t.translation.x = target[0]
        self.ros_t.translation.y = target[1]
        self.ros_t.translation.z = target[2]

        self.cur = target
        self.pub_traj.publish(self.cart_traj_point)
        self.rate.sleep()

        while timeq < 1.0:
            timeq = timeq + quadT
            if (timeq > 1.0):
                timeq = 1.0
            q = pyquaternion.Quaternion.slerp(self.cur_quat, target_quat, timeq)
            self.ros_t.rotation.x = q[1]
            self.ros_t.rotation.y = q[2]
            self.ros_t.rotation.z = q[3]
            self.ros_t.rotation.w = q[0]
            self.pub_traj.publish(self.cart_traj_point)
            self.rate.sleep()

        self.ros_t.translation.x = target[0]
        self.ros_t.translation.y = target[1]
        self.ros_t.translation.z = target[2]
        self.ros_t.rotation.x = target_quat[1]
        self.ros_t.rotation.y = target_quat[2]
        self.ros_t.rotation.z = target_quat[3]
        self.ros_t.rotation.w = target_quat[0]
        self.cur_quat = target_quat
        self.cur = target
        self.pub_traj.publish(self.cart_traj_point)
        self.rate.sleep()

    def moveGuarded(self, cart_direction=[0,0,0], step_width_in_m=0.00001, max_force_in_n=40.0):
        # Quaternion(w, x, y, z)
        dire = np.array(cart_direction)
        step = dire * step_width_in_m
        # If no wrench reading return
        self.lock_wrench.acquire()
        self.wrench_data_internal = self.wrench_data
        self.lock_wrench.release()
        if self.wrench_data_internal == None:
            print("Returning, no wrench reading!")
            return
        
        force = np.array([self.wrench_data_internal.force.x, self.wrench_data_internal.force.y, self.wrench_data_internal.force.z])
        force_length = np.linalg.norm(force * dire)

        print("Start Guarded Move")
        # add time-based smoothing?
        while force_length < max_force_in_n:
            self.cur = self.cur + step
            self.ros_t.translation.x = self.cur[0]
            self.ros_t.translation.y = self.cur[1]
            self.ros_t.translation.z = self.cur[2]
            self.pub_traj.publish(self.cart_traj_point)
            self.rate.sleep()

            self.lock_wrench.acquire()
            self.wrench_data_internal = self.wrench_data
            self.lock_wrench.release()
            force = np.array([self.wrench_data_internal.force.x, self.wrench_data_internal.force.y, self.wrench_data_internal.force.z])
            force_length = np.linalg.norm(force * dire)

        print("Force Sensed!")

    def normalized(self,a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a))
        # l2[l2==0] = 1
        return a / l2

if __name__ == "__main__":
    c = GraspTest()