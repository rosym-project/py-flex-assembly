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

from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

import threading

import pyquaternion

import numpy as np

import time

import serial

class ClampIt(object):
    def __init__(self):
        self.use_gripper = True
        self.skip_first_phase = False
        self.real = False

        if self.real:
            self.ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=2,)

        # # TEST GRIPPER
        # rospy.wait_for_service('/gripper1/open_gripper')
        # try:
        #     open_g = rospy.ServiceProxy('/gripper1/open_gripper', Empty)
        #     open_g()
        # except rospy.ServiceException as e:
        #     print("Service call failed: %s"%e)
        # return
        # # TEST GRIPPER

        self.clampPose = Pose()
        self.clampPose.position.z = 0.285982
        # Below needs to be filled by the vision
        self.clampPose.position.x = -0.22
        self.clampPose.position.y = -0.44
        self.clampPose.orientation.w = 1
        self.clampPose.orientation.x = 0
        self.clampPose.orientation.y = 0
        self.clampPose.orientation.z = 0
        self.clampQuat = pyquaternion.Quaternion(w=self.clampPose.orientation.w,x=self.clampPose.orientation.x,y=self.clampPose.orientation.y,z=self.clampPose.orientation.z) * pyquaternion.Quaternion(axis=[0, 1, 0], angle=180.0 / 180.0 * 3.14159265)

        m = MoveRequest()
        p = Pose()
        p.position.x = 0.000158222
        p.position.y = -0.675439
        p.position.z = 0.285982
        p.orientation.w = 0
        p.orientation.x = 0
        p.orientation.y = 1
        p.orientation.z = 0

        rospy.init_node('coordcc', anonymous=False)

        if not self.skip_first_phase:
            # if self.use_gripper:
            #     if self.real:
            #         self.open_gripper()
            #     else:
            #         rospy.wait_for_service('/gripper1/open_gripper')
            #         try:
            #             open_g = rospy.ServiceProxy('/gripper1/open_gripper', Empty)
            #             open_g()
            #         except rospy.ServiceException as e:
            #             print("Service call failed: %s"%e)
            #     time.sleep(1)
            
            rospy.wait_for_service('/css/move_srv')
            time.sleep(1)
            try:
                add_two_ints = rospy.ServiceProxy('/css/move_srv', Move)

                # p.position.x = self.clampPose.position.x
                # p.position.y = self.clampPose.position.y
                # p.position.z = 0.1

                # p.orientation.w = self.clampQuat[0]
                # p.orientation.x = self.clampQuat[1]
                # p.orientation.y = self.clampQuat[2]
                # p.orientation.z = self.clampQuat[3]

                # p.position.x = -0.258
                # p.position.y = -0.53
                # p.position.z = 0.049

                # p.position.x = -0.21
                # p.position.y = -0.53
                # p.position.z = 0.049

                # self.clampQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)

                self.clampQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)

                self.clampQuat = self.clampQuat * pyquaternion.Quaternion(axis=[1, -1, 0], angle=15 / 180.0 * 3.14159265)


                p.orientation.x = self.clampQuat[1]
                p.orientation.y = self.clampQuat[2]
                p.orientation.z = self.clampQuat[3]
                p.orientation.w = self.clampQuat[0]

                ####################################################################################################
                # Move 10) Push Down
                print("Phase #10: Push Down!")
                p.position.x = -0.258
                p.position.y = -0.53
                p.position.z = 0.046
                # self.outQuat = self.clamp1Quat * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=15 / 180.0 * 3.14159265)
                # p.orientation.x = self.outQuat[1]
                # p.orientation.y = self.outQuat[2]
                # p.orientation.z = self.outQuat[3]
                # p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 10.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #10")

                return

                time.sleep(3)

                # 1) Rotate init
                p.position.x = -0.0452333
                p.position.y = -0.594722
                p.position.z = 0.301398
                self.clampQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                p.orientation.x = self.clampQuat[1]
                p.orientation.y = self.clampQuat[2]
                p.orientation.z = self.clampQuat[3]
                p.orientation.w = self.clampQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 10.0
                m.i_max_rot_sec = 20.0
                resp1 = add_two_ints(m)
                print("1 done")
                time.sleep(5)

                # 2) Move to lookout
                p.position.x = -0.25
                p.position.y = -0.594722
                p.position.z = 0.45
                self.clampQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                p.orientation.x = self.clampQuat[1]
                p.orientation.y = self.clampQuat[2]
                p.orientation.z = self.clampQuat[3]
                p.orientation.w = self.clampQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 10.0
                m.i_max_rot_sec = 20.0
                resp1 = add_two_ints(m)
                print("2 done")
                time.sleep(1)

                return

                # 2) Move clamp 1







                p.position.z = 0.02

                m.i_pose = p
                m.i_max_trans_sec = 60.0
                m.i_max_rot_sec = 5.0
                resp1 = add_two_ints(m)

                print("2 done")

                if self.use_gripper:
                    if self.real:
                        self.close_gripper()
                    else:
                        rospy.wait_for_service('/gripper1/close_gripper')
                        try:
                            close_g = rospy.ServiceProxy('/gripper1/close_gripper', Empty)
                            close_g()
                        except rospy.ServiceException as e:
                            print("Service call failed: %s"%e)
                    time.sleep(1)

                p.position.z = 0.1

                m.i_pose = p
                m.i_max_trans_sec = 60.0
                m.i_max_rot_sec = 5.0
                resp1 = add_two_ints(m)
                time.sleep(1)

                print("3 done")

                p.position.x = 0.0
                p.position.y = -0.59
                p.position.z = 0.1

                p.orientation.w = 0
                p.orientation.x = 0
                p.orientation.y = -0.985
                p.orientation.z = -0.172

                m.i_pose = p
                m.i_max_trans_sec = 15.0
                m.i_max_rot_sec = 5.0
                resp1 = add_two_ints(m)
                time.sleep(1)
                print("4 done")

                print(resp1)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
        
        rospy.wait_for_service('/css/move_srv')
        try:
            add_two_ints = rospy.ServiceProxy('/css/move_srv', Move)

            p.position.x = 0.0
            p.position.y = -0.59
            p.position.z = 0.04

            p.orientation.w = 0
            p.orientation.x = 0
            p.orientation.y = -0.985
            p.orientation.z = -0.172

            m.i_pose = p
            m.i_max_trans_sec = 100.0
            m.i_max_rot_sec = 5.0
            resp1 = add_two_ints(m)
            time.sleep(1)
            print("5 done")


            # p.position.y = -0.56

            # m.i_pose = p
            # m.i_max_trans_sec = 100.0
            # m.i_max_rot_sec = 5.0
            # resp1 = add_two_ints(m)
            # time.sleep(1)
            # print("6 done")

            # COMPLIANT IN Y AND PUSH IN Y = Contact Constraint Y

            print(resp1)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        #####################
        print("END")

    def close_gripper(self):
        self.ser.open()
        self.ser.write([int('00000011', 2)])
        time.sleep(0.01)
        self.ser.write([int('00000001', 2)])
        self.ser.close()

    def open_gripper(self):
        self.ser.open()
        self.ser.write([int('00000011', 2)])
        time.sleep(0.01)
        self.ser.write([int('00000010', 2)])
        self.ser.close()

if __name__ == "__main__":
    c = ClampIt()