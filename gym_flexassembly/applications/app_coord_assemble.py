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

        self.clamp1 = Pose()
        self.clamp1.position.z = 0.049
        # Below needs to be filled by the vision
        self.clamp1.position.x = -0.348
        self.clamp1.position.y = -0.656
        self.clamp1.orientation.w = 0
        self.clamp1.orientation.x = 0
        self.clamp1.orientation.y = 1
        self.clamp1.orientation.z = 0
        # self.clamp1Quat = pyquaternion.Quaternion(w=self.clamp1.orientation.w,x=self.clamp1.orientation.x,y=self.clamp1.orientation.y,z=self.clamp1.orientation.z) * pyquaternion.Quaternion(axis=[0, 1, 0], angle=180.0 / 180.0 * 3.14159265)
        self.clamp1Quat = pyquaternion.Quaternion(w=self.clamp1.orientation.w,x=self.clamp1.orientation.x,y=self.clamp1.orientation.y,z=self.clamp1.orientation.z)

        m = MoveRequest()
        p = Pose()
        p.position.x = -0.0452333
        p.position.y = -0.594722
        p.position.z = 0.301398
        # TODO init
        self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=0.0 / 180.0 * 3.14159265)
        p.orientation.x = self.outQuat[1]
        p.orientation.y = self.outQuat[2]
        p.orientation.z = self.outQuat[3]
        p.orientation.w = self.outQuat[0]

        rospy.init_node('coordcc', anonymous=False)

        if not self.skip_first_phase:
            # Open Gripper
            if self.use_gripper:
                if self.real:
                    self.open_gripper()
                else:
                    rospy.wait_for_service('/gripper1/open_gripper')
                    try:
                        open_g = rospy.ServiceProxy('/gripper1/open_gripper', Empty)
                        open_g()
                    except rospy.ServiceException as e:
                        print("Service call failed: %s"%e)
                time.sleep(1)
            
            # Wait for movement service
            rospy.wait_for_service('/css/move_srv')
            time.sleep(1)
            try:
                add_two_ints = rospy.ServiceProxy('/css/move_srv', Move)

                # Move 1) Feed same position for initialization
                print("Phase #1: Feed back position to initialization!")
                p.position.x = -0.0452333
                p.position.y = -0.594722
                p.position.z = 0.301398
                self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=0.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 30.0
                m.i_max_rot_sec = 20.0
                resp1 = add_two_ints(m)
                print("Done with Phase #1")

                time.sleep(2)

                # Move 2) Rotate to initial observer pose
                print("Phase #2: Rotate to initial observer pose!")
                p.position.x = -0.0452333
                p.position.y = -0.594722
                p.position.z = 0.301398
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 30.0
                m.i_max_rot_sec = 20.0
                resp1 = add_two_ints(m)
                print("Done with Phase #2")
                time.sleep(2)

                # Move 3) Move to observer pose
                print("Phase #3: Move to observer pose!")
                p.position.x = -0.25
                p.position.y = -0.594722
                p.position.z = 0.45
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 10.0
                m.i_max_rot_sec = 20.0
                resp1 = add_two_ints(m)
                print("Done with Phase #3")
                time.sleep(1)

                # Move 4) Move to pre grasp clamp 1
                print("Phase #4: Move to pre grasp clamp 1!")
                p.position.x = self.clamp1.position.x
                p.position.y = self.clamp1.position.y
                p.position.z = 0.15
                self.outQuat = self.clamp1Quat * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 30.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #4")
                time.sleep(1)

                # Move 5) Move to grasp clamp 1
                print("Phase #5: Move to grasp clamp 1!")
                p.position.x = self.clamp1.position.x
                p.position.y = self.clamp1.position.y
                p.position.z = 0.049
                self.outQuat = self.clamp1Quat * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 80.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #5")
                time.sleep(1)

                # Close Gripper
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


                # Move 6) Move back to pre grasp clamp 1
                print("Phase #6: Move back to pre grasp clamp 1!")
                p.position.x = self.clamp1.position.x
                p.position.y = self.clamp1.position.y
                p.position.z = 0.15
                self.outQuat = self.clamp1Quat * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 60.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #6")
                time.sleep(1)

                # Move 7) Move to rail drag pose
                print("Phase #7: Move to rail drag pose!")
                p.position.x = -0.258
                p.position.y = -0.56
                p.position.z = 0.15
                self.outQuat = self.clamp1Quat * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=15 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 30.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #7")
                time.sleep(1)

                # Move 8) Move Down
                print("Phase #8: Move Down!")
                p.position.x = -0.258
                p.position.y = -0.56
                p.position.z = 0.049
                self.outQuat = self.clamp1Quat * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=15 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 80.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #8")
                time.sleep(1)

                # Move 9) Move straight to Rail
                print("Phase #9: Move straight to Rail!")
                p.position.x = -0.258
                p.position.y = -0.53
                p.position.z = 0.049
                self.outQuat = self.clamp1Quat * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=15 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 90.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #9")
                time.sleep(1)
                
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
        
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