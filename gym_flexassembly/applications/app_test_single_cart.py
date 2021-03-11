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

class SingleCart(object):
    def __init__(self):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('coord', anonymous=False)
        print("Init coord node")

        # write trajectory command
        self.pub_traj = rospy.Publisher("/cart/traj_setpoint", MultiDOFJointTrajectoryPoint, queue_size=1,latch=True)

        # Wait for the first pose feedback
        self.rate = rospy.Rate(500) # 10hz

        self.rate.sleep()

        self.cart_traj_point = MultiDOFJointTrajectoryPoint()
        self.ros_t = Transform()
        # self.ros_t.translation.x = -0.0285429
        # self.ros_t.translation.y = -0.529713
        # self.ros_t.translation.z = 0.467274
        self.ros_t.translation.x = -0.0285429
        self.ros_t.translation.y = -0.529713
        self.ros_t.translation.z = 0.46
        self.cur = np.array([self.ros_t.translation.x, self.ros_t.translation.y, self.ros_t.translation.z])

        self.ros_t.rotation.w = 0.0174733
        self.ros_t.rotation.x = 0.68765
        self.ros_t.rotation.y = -0.725645
        self.ros_t.rotation.z = 0.0164777
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

        # -0.0277058
        # -0.532249 
        # 0.321759  
        # 0.0440427 
        # -0.687174 
        # 0.723958  
        # 0.0416703

        time.sleep(1)

        for i in range(250):
            self.ros_t.translation.z = self.ros_t.translation.z - 0.001
            self.pub_traj.publish(self.cart_traj_point)
            time.sleep(0.1)

        time.sleep(4)

        for i in range(250):
            self.ros_t.translation.z = self.ros_t.translation.z + 0.001
            self.pub_traj.publish(self.cart_traj_point)
            time.sleep(0.1)


        return
        
        speed = 0.0001
        solo_tor_speed = 0.00005

        # target_quat_left = pyquaternion.Quaternion(w=0.707,x=0,y=0.707,z=0) * pyquaternion.Quaternion(axis=[0, 1, 0], angle=25.0 / 180.0 * 3.14159265)
        # target_quat_left = pyquaternion.Quaternion(w=0.707,x=0,y=0.707,z=0)

        self.moveTo(target=np.array([-0.0285429,-0.529713,0.46]), target_quat=self.cur_quat, step_width=speed, solo_rot_step=solo_tor_speed)
        time.sleep(0.5)

        # self.moveTo(target=np.array([-0.0285429,-0.529713,0.6]), target_quat=self.cur_quat, step_width=speed, solo_rot_step=solo_tor_speed)
        # time.sleep(0.5)

        self.moveTo(target=np.array([-0.0285429,-0.529713,0.1]), target_quat=self.cur_quat, step_width=speed, solo_rot_step=solo_tor_speed)
        time.sleep(0.5)

        self.moveTo(target=np.array([-0.0285429,-0.529713,0.46]), target_quat=self.cur_quat, step_width=speed, solo_rot_step=solo_tor_speed)
        time.sleep(0.5)

        # self.moveTo(target=np.array([0.6,0.0,0.7]), target_quat=(target_quat_left * pyquaternion.Quaternion(axis=[0, 1, 0], angle=25.0 / 180.0 * 3.14159265)), step_width=speed, solo_rot_step=solo_tor_speed)

        # self.moveTo(target=np.array([0.8,0.0,0.5]), target_quat=(target_quat_left * pyquaternion.Quaternion(axis=[0, 1, 0], angle=25.0 / 180.0 * 3.14159265)), step_width=speed, solo_rot_step=solo_tor_speed)
        
        print("Finished")


    def moveTo(self, target, target_quat, step_width=0.0001, solo_rot_step=0.01):
        print("Innn")
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

    def normalized(self,a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a))
        # l2[l2==0] = 1
        return a / l2

if __name__ == "__main__":
    c = SingleCart()