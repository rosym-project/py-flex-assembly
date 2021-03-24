import os

import time

import signal

import sys

import math

import rospy

from geometry_msgs.msg import Pose

from cosima_msgs.srv import GetClamp
from cosima_msgs.srv import Move
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
            p = Pose()
            p.position.x = 0.000158222
            p.position.y = -0.675439
            p.position.z = 0.285982

            p.orientation.w = 0
            p.orientation.x = 0
            p.orientation.y = 1
            p.orientation.z = 0

            resp1 = add_two_ints(p)

            time.sleep(4)

            p.position.z = 0.25
            p.orientation.w = 0
            p.orientation.x = -0.369
            p.orientation.y = -0.929
            p.orientation.z = 0

            resp1 = add_two_ints(p)

            time.sleep(4)

            p.position.z = 0.285982
            p.orientation.w = 0
            p.orientation.x = 0
            p.orientation.y = 1
            p.orientation.z = 0

            resp1 = add_two_ints(p)

            time.sleep(4)


            print(resp1)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(4)
        print("END")


if __name__ == "__main__":
    c = TableWiping()