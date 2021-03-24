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

import threading

import numpy as np

import time

class TableWiping(object):
    def __init__(self):
        rospy.init_node('coordcc', anonymous=False)
        print("YAYYYYY")
        rospy.wait_for_service('/css/move_srv')
        time.sleep(1)
        try:
            add_two_ints = rospy.ServiceProxy('/css/move_srv', Move)
            m = MoveRequest()
            
            p = Pose()
            p.position.x = 0.000158222
            p.position.y = -0.675439
            p.position.z = 0.285982

            p.orientation.w = 0
            p.orientation.x = 0
            p.orientation.y = 1
            p.orientation.z = 0

            m.i_pose = p
            m.i_max_trans_sec = 15.0
            m.i_max_rot_sec = 5.0
            resp1 = add_two_ints(m)

            print("1 done")

            time.sleep(1)

            p.position.z = 0.12
            p.orientation.w = 0
            p.orientation.x = -0.369
            p.orientation.y = -0.929
            p.orientation.z = 0

            m.i_pose = p
            m.i_max_trans_sec = 15.0
            m.i_max_rot_sec = 5.0
            resp1 = add_two_ints(m)

            print("2 done")

            time.sleep(0.5)

            p.position.z = 0.09
            p.orientation.w = 0
            p.orientation.x = -0.369
            p.orientation.y = -0.929
            p.orientation.z = 0

            m.i_pose = p
            m.i_max_trans_sec = 60.0
            m.i_max_rot_sec = 5.0
            resp1 = add_two_ints(m)

            print("3 done")

            time.sleep(1)

            p.position.z = 0.285982
            p.orientation.w = 0
            p.orientation.x = 0
            p.orientation.y = 1
            p.orientation.z = 0

            m.i_pose = p
            m.i_max_trans_sec = 7.0
            m.i_max_rot_sec = 5.0
            resp1 = add_two_ints(m)

            print("4 done")

            time.sleep(1)


            print(resp1)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(4)
        print("END")


if __name__ == "__main__":
    c = TableWiping()