"""Table Wiping Scenario for CMCI.
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

from std_msgs.msg import Float32MultiArray

from geometry_msgs.msg import Pose
# from geometry_msgs.msg import Point
# from geometry_msgs.msg import Quaternion

import threading

import numpy as np

class TableWiping(object):
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
        rospy.Subscriber("/cart_imped_high/cartesian_pose", Pose, self.listener_cur_ee_pose)
        self.var_cur_ee_pose = Pose()
        self.lock = threading.Lock()

        # write trajectory command
        self.pub_traj = rospy.Publisher("/traj", Pose, queue_size=1,latch=True)

        # write trajectory command pass through without interpolation
        self.pub_traj_pt = rospy.Publisher("/traj_pt", Pose, queue_size=1,latch=True)

        # # spin() simply keeps python from exiting until this node is stopped
        # rospy.spin()
        centerX = -0.3
        centerY = 0.5
        centerZ = 0.46

        # Goto home pos
        goal = Pose()
        goal.position.x = centerX
        goal.position.y = centerY
        goal.position.z = centerZ + 0.12

        goal.orientation.x = 1
        goal.orientation.y = 0
        goal.orientation.z = 0
        goal.orientation.w = 0

        self.pub_traj.publish(goal)

        print("Send (Over Table) " + str(goal))

        self.rate = rospy.Rate(10) # 10hz
        self.rate_circ = rospy.Rate(500) # 500hz
        cur_ee_pose_tmp = Pose()
        # Wait until reached

        start_time = time.time_ns() / (10 ** 9) # convert to floating-point seconds
        while (time.time_ns() / (10 ** 9) - start_time) < 20.0:
            with self.lock:
                cur_ee_pose_tmp = self.var_cur_ee_pose
#             print (math.fabs(
# np.linalg.norm(np.array([cur_ee_pose_tmp.position.x,cur_ee_pose_tmp.position.y,cur_ee_pose_tmp.position.z])-np.array([goal.position.x,goal.position.y,goal.position.z]))))
            if math.fabs(
np.linalg.norm(np.array([cur_ee_pose_tmp.position.x,cur_ee_pose_tmp.position.y,cur_ee_pose_tmp.position.z])-np.array([goal.position.x,goal.position.y,goal.position.z]))) < 0.01:
                # reached
                break
            self.rate.sleep()

        # GO down
        goal.position.z = centerZ
        self.pub_traj.publish(goal)

        print("Send (Down) " + str(goal))

        # Wait until reached
        start_time = time.time_ns() / (10 ** 9) # convert to floating-point seconds
        while (time.time_ns() / (10 ** 9) - start_time) < 20.0:
            with self.lock:
                cur_ee_pose_tmp = self.var_cur_ee_pose
#             print (math.fabs(
# np.linalg.norm(np.array([cur_ee_pose_tmp.position.x,cur_ee_pose_tmp.position.y,cur_ee_pose_tmp.position.z])-np.array([goal.position.x,goal.position.y,goal.position.z]))))
            if math.fabs(
np.linalg.norm(np.array([cur_ee_pose_tmp.position.x,cur_ee_pose_tmp.position.y,cur_ee_pose_tmp.position.z])-np.array([goal.position.x,goal.position.y,goal.position.z]))) < 0.01:
                # reached
                break
            self.rate.sleep()

        # ACTIVATE FF

        # CIRCULAR Trajectory
        # board parallel to floor
        boardAngle_deg = 0
        boardAngle_rad = boardAngle_deg / 360.0 * 2.0 * math.pi
        BoardRot = np.array([[math.cos(boardAngle_rad),0,math.sin(boardAngle_rad)],
                             [0,1,0],
                             [-math.sin(boardAngle_rad),0,math.cos(boardAngle_rad)]])

        _timescale = 0.6
        radius = 0.1

        print("Start Circle")

        start_time = time.time_ns() / (10 ** 9) # convert to floating-point seconds
        # while (time.time_ns() / (10 ** 9) - start_time) < 10.0:
        while True:
            # circ = BoardRot * np.transpose(np.array([radius * math.cos(_timescale * (time.time_ns() / (10 ** 9) - start_time)), radius * math.sin(_timescale * (time.time_ns() / (10 ** 9) - start_time)), 0.0])) + np.transpose(np.array([centerX, centerY, centerZ]))
            circ = np.array([radius * math.cos(_timescale * (time.time_ns() / (10 ** 9) - start_time)), radius * math.sin(_timescale * (time.time_ns() / (10 ** 9) - start_time)), 0.0]) + np.array([centerX, centerY, centerZ])

            goal.position.x = circ[0]
            goal.position.y = circ[1]
            goal.position.z = circ[2]

            self.pub_traj_pt.publish(goal)

            print(goal)

            self.rate_circ.sleep()

        # rate = rospy.Rate(10) # 10hz
        # while not rospy.is_shutdown():
        #     hello_str = "hello world %s" % rospy.get_time()
        #     rospy.loginfo(hello_str)
        #     pub.publish(hello_str)
        #     rate.sleep()

        print("Spin = End")
        rospy.spin()

    def listener_cur_ee_pose(self, payload):
        with self.lock:
            # data = np.array(payload.data).reshape((4, 4))
            # self.var_cur_ee_pose.position.x = data[3,0]
            # self.var_cur_ee_pose.position.y = data[3,1]
            # self.var_cur_ee_pose.position.z = data[3,2]

            self.var_cur_ee_pose = payload
            # print(self.var_cur_ee_pose.position)
            
            # TODO etc...
            # self.var_cur_ee_pose.orientation.x = data[3]
            # self.var_cur_ee_pose.orientation.y = data[4]
            # self.var_cur_ee_pose.orientation.z = data[5]
            # self.var_cur_ee_pose.orientation.w = data[6]

if __name__ == "__main__":
    # try:
    #     # TableWiping()
    # except rospy.ROSInterruptException:
    #     pass
    c = TableWiping()