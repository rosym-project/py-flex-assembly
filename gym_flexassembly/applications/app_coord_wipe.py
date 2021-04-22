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

class WipeIt(object):
    def __init__(self):
        m = MoveRequest()
        p = Pose()
        p.position.x = -0.0452333
        p.position.y = -0.594722
        p.position.z = 0.301398

        self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=0.0 / 180.0 * 3.14159265)
        p.orientation.x = self.outQuat[1]
        p.orientation.y = self.outQuat[2]
        p.orientation.z = self.outQuat[3]
        p.orientation.w = self.outQuat[0]

        rospy.init_node('coordcc', anonymous=False)

        # receive ext wrench
        rospy.Subscriber("/robot/wrench", Wrench, self.listener_ft_wrench)
        self.wrench_data = None
        self.wrench_data_internal = None
        self.lock_wrench = threading.Lock()


        
        # Wait for movement service
        rospy.wait_for_service('/css/move_srv')
        time.sleep(1)
        try:
            add_two_ints = rospy.ServiceProxy('/css/move_srv', Move)

            # # Move 1) Feed same position for initialization
            # print("Phase #1: Feed back position to initialization!")
            # p.position.x = 0.0
            # p.position.y = -0.594722
            # p.position.z = 0.42
            # self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[-1, 1, 0], angle=-89.0 / 180.0 * 3.14159265)
            # p.orientation.x = self.outQuat[1]
            # p.orientation.y = self.outQuat[2]
            # p.orientation.z = self.outQuat[3]
            # p.orientation.w = self.outQuat[0]
            # m.i_pose = p
            # m.i_max_trans_sec = 60.0
            # m.i_max_rot_sec = 40.0
            # resp1 = add_two_ints(m)
            # print("Done with Phase #1")

            # # Move 2) Feed same position for initialization
            # print("Phase #1: Feed back position to initialization!")
            # p.position.x = 0.57
            # p.position.y = -0.594722
            # p.position.z = 0.42
            # self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[-1, 1, 0], angle=-89.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=35.0/ 180.0 * 3.14159265)
            # p.orientation.x = self.outQuat[1]
            # p.orientation.y = self.outQuat[2]
            # p.orientation.z = self.outQuat[3]
            # p.orientation.w = self.outQuat[0]
            # m.i_pose = p
            # m.i_max_trans_sec = 50.0
            # m.i_max_rot_sec = 60.0
            # resp1 = add_two_ints(m)
            # print("Done with Phase #2")

            # time.sleep(2)

            # rospy.wait_for_service('/css/updateContactSituationBlocking_srv')
            # try:
            #     switchCS = rospy.ServiceProxy('/css/updateContactSituationBlocking_srv', ContactSituation)
            #     c = ContactSituationRequest()
            #     c.kp_trans.x = 1600.0
            #     c.kp_trans.y = 1600.0
            #     c.kp_trans.z = 1600.0
            #     c.kp_rot.x = 300.0
            #     c.kp_rot.y = 300.0
            #     c.kp_rot.z = 100.0

            #     c.kd_trans.x = 80.0
            #     c.kd_trans.y = 80.0
            #     c.kd_trans.z = 80.0
            #     c.kd_rot.x = 1.2
            #     c.kd_rot.y = 1.2
            #     c.kd_rot.z = 1.0

            #     c.fdir_trans.x = 0.0
            #     c.fdir_trans.y = 0.0
            #     c.fdir_trans.z = 0.0
            #     c.fdir_rot.x = 0.0
            #     c.fdir_rot.y = 0.0
            #     c.fdir_rot.z = 0.0

            #     c.force_trans.x = 0.0
            #     c.force_trans.y = 0.0
            #     c.force_trans.z = 0.0
            #     c.force_rot.x = 0.0
            #     c.force_rot.y = 0.0
            #     c.force_rot.z = 0.0

            #     c.time = 3.0

            #     c.update_pose_first = True

            #     switchCS(c)
            # except rospy.ServiceException as e:
            #     print("Service call failed: %s"%e)

            up = True
            for i in range(10):
                # Move 3) Feed same position for initialization
                print("Phase #1: Feed back position to initialization!")
                p.position.x = 0.57
                p.position.y = -0.594722
                if up:
                    p.position.z = 0.6
                    up = False
                else:
                    p.position.z = 0.3
                    up = True
                self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[-1, 1, 0], angle=-89.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=35.0/ 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 15.0
                m.i_max_rot_sec = 60.0
                resp1 = add_two_ints(m)
                print("Done with Phase #3")

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        print("END")

    def listener_ft_wrench(self, data):
        self.lock_wrench.acquire()
        self.wrench_data = data
        self.lock_wrench.release()

if __name__ == "__main__":
    c = WipeIt()