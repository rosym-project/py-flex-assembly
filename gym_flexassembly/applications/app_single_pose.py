import os

import time

import signal

import sys

import math

import rospy

from geometry_msgs.msg import Pose

from cosima_msgs.srv import GetClamp
from cosima_msgs.srv import Move, MoveResponse, MoveRequest
from cosima_msgs.srv import Assemble, AssembleResponse
from cosima_msgs.srv import ContactSituation, ContactSituationResponse, ContactSituationRequest
from cosima_msgs.srv import ContactForce, ContactForceResponse

from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Wrench

from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

import threading

import pyquaternion

import numpy as np

import time

import serial

class SinglePose(object):
    def __init__(self):
        self.use_gripper = True
        self.skip_first_phase = False
        self.real = False


        m = MoveRequest()
        p = Pose()
        p.position.x = -0.25
        p.position.y = -0.594722
        p.position.z = 0.35
        self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=0.0 / 180.0 * 3.14159265)
        p.orientation.x = self.outQuat[1]
        p.orientation.y = self.outQuat[2]
        p.orientation.z = self.outQuat[3]
        p.orientation.w = self.outQuat[0]

        rospy.init_node('coordcc', anonymous=False)

        # Wait for movement service
        rospy.wait_for_service('/css/move_srv')
        rospy.wait_for_service('pose_estimation')
        time.sleep(1)
        try:
            
            add_two_ints = rospy.ServiceProxy('/css/move_srv', Move)

            # Move 1) Feed same position for initialization
            print("Phase #1: Feed back position to initialization!")


            p.position.x = -0.258
            p.position.y = -0.56
            p.position.z = 0.1
            self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=-15 / 180.0 * 3.14159265)
            p.orientation.x = self.outQuat[1]
            p.orientation.y = self.outQuat[2]
            p.orientation.z = self.outQuat[3]
            p.orientation.w = self.outQuat[0]

            m.i_pose = p
            m.i_max_trans_sec = 30.0
            m.i_max_rot_sec = 30.0
            resp1 = add_two_ints(m)

            return

            time.sleep(2)

            print("GET IMAGE")
            estimate = rospy.ServiceProxy('pose_estimation', GetClamp)
            estimation = estimate().i_pose
            pos = estimation.position
            pos = np.array([pos.x, pos.y, pos.z])
            _pos = pos * 1000
            print(f'Pos [{_pos[0]:.2f}, {_pos[1]:.2f}, {_pos[2]:.2f}]')

            orn = estimation.orientation
            orn = [orn.x, orn.y, orn.z, orn.w]


            p.position.x = estimation.position.x
            p.position.y = estimation.position.y
            p.position.z = 0.1
            p.orientation = estimation.orientation

            m.i_pose = p
            m.i_max_trans_sec = 20.0
            m.i_max_rot_sec = 20.0
            resp1 = add_two_ints(m)

            time.sleep(10)

            p.position.z = 0.065
            m.i_pose = p
            m.i_max_trans_sec = 30.0
            m.i_max_rot_sec = 30.0
            resp1 = add_two_ints(m)

            time.sleep(1)

            rospy.wait_for_service('/gripper1/close_gripper')
            try:
                close_g = rospy.ServiceProxy('/gripper1/close_gripper', Empty)
                close_g()
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

            time.sleep(10)

            rospy.wait_for_service('/gripper1/open_gripper')
            try:
                open_g = rospy.ServiceProxy('/gripper1/open_gripper', Empty)
                open_g()
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

            time.sleep(0.5)

            p.position.z = 0.1
            m.i_pose = p
            m.i_max_trans_sec = 30.0
            m.i_max_rot_sec = 30.0
            resp1 = add_two_ints(m)


        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        print("END")

if __name__ == "__main__":
    c = SinglePose()