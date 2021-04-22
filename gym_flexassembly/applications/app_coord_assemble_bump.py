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
        # self.clamp1.position.z = 0.049
        # # Below needs to be filled by the vision
        # self.clamp1.position.x = -0.348
        # self.clamp1.position.y = -0.656

        self.clamp1.position.z = 0.065
        # Below needs to be filled by the vision
        self.clamp1.position.x = -0.335
        self.clamp1.position.y = -0.48

        self.clamp1.orientation.w = 0
        self.clamp1.orientation.x = 0
        self.clamp1.orientation.y = 1
        self.clamp1.orientation.z = 0
        # self.clamp1Quat = pyquaternion.Quaternion(w=self.clamp1.orientation.w,x=self.clamp1.orientation.x,y=self.clamp1.orientation.y,z=self.clamp1.orientation.z) * pyquaternion.Quaternion(axis=[0, 1, 0], angle=180.0 / 180.0 * 3.14159265)
        self.clamp1Quat = pyquaternion.Quaternion(w=self.clamp1.orientation.w,x=self.clamp1.orientation.x,y=self.clamp1.orientation.y,z=self.clamp1.orientation.z) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265)

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

        # receive ext wrench
        rospy.Subscriber("/robot/wrench", Wrench, self.listener_ft_wrench)
        self.wrench_data = None
        self.wrench_data_internal = None
        self.lock_wrench = threading.Lock()

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

                # # Move 1) Feed same position for initialization
                # print("Phase #1: Feed back position to initialization!")
                # p.position.x = -0.0452333
                # p.position.y = -0.594722
                # p.position.z = 0.301398
                # self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=0.0 / 180.0 * 3.14159265)
                # p.orientation.x = self.outQuat[1]
                # p.orientation.y = self.outQuat[2]
                # p.orientation.z = self.outQuat[3]
                # p.orientation.w = self.outQuat[0]
                # m.i_pose = p
                # m.i_max_trans_sec = 30.0
                # m.i_max_rot_sec = 20.0
                # resp1 = add_two_ints(m)
                # print("Done with Phase #1")

                # time.sleep(2)

                # # Move 2) Rotate to initial observer pose
                # print("Phase #2: Rotate to initial observer pose!")
                # p.position.x = -0.0452333
                # p.position.y = -0.594722
                # p.position.z = 0.301398
                # self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                # p.orientation.x = self.outQuat[1]
                # p.orientation.y = self.outQuat[2]
                # p.orientation.z = self.outQuat[3]
                # p.orientation.w = self.outQuat[0]
                # m.i_pose = p
                # m.i_max_trans_sec = 8.0
                # m.i_max_rot_sec = 10.0
                # resp1 = add_two_ints(m)
                # print("Done with Phase #2")
                # time.sleep(2)

                # Move 3) Move to observer pose
                print("Phase #3: Move to observer pose!")
                p.position.x = -0.25
                p.position.y = -0.594722
                p.position.z = 0.45
                self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=0.0 / 180.0 * 3.14159265)
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

                print("GET IMAGE")
                estimate = rospy.ServiceProxy('pose_estimation', GetClamp)
                estimation = estimate().i_pose
                pos = estimation.position
                pos = np.array([pos.x, pos.y, pos.z])
                _pos = pos * 1000
                print(f'Pos [{_pos[0]:.2f}, {_pos[1]:.2f}, {_pos[2]:.2f}]')
                self.clamp1.position = estimation.position
                self.clamp1.orientation = estimation.orientation
                self.clamp1Quat = pyquaternion.Quaternion(w=self.clamp1.orientation.w,x=self.clamp1.orientation.x,y=self.clamp1.orientation.y,z=self.clamp1.orientation.z)

                # Move 4) Move to pre grasp clamp 1
                print("Phase #4: Move to pre grasp clamp 1!")
                p.position.x = self.clamp1.position.x
                p.position.y = self.clamp1.position.y
                p.position.z = 0.1
                self.outQuat = self.clamp1Quat
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 10.0
                m.i_max_rot_sec = 20.0
                resp1 = add_two_ints(m)
                print("Done with Phase #4")
                time.sleep(1)

                # Move 5) Move to grasp clamp 1
                print("Phase #5: Move to grasp clamp 1!")
                p.position.x = self.clamp1.position.x
                p.position.y = self.clamp1.position.y
                p.position.z = 0.065
                self.outQuat = self.clamp1Quat
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 50.0
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
                p.position.z = 0.24
                self.outQuat = self.clamp1Quat
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 20.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #6")
                time.sleep(1)

                # Move 7) Move to rail drag pose
                print("Phase #7: Move to rail drag pose!")
                p.position.x = -0.258
                p.position.y = -0.56
                p.position.z = 0.1
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=-15 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 20.0
                m.i_max_rot_sec = 20.0
                resp1 = add_two_ints(m)
                print("Done with Phase #7")
                time.sleep(1)

                # Move 8) Move Down
                print("Phase #8: Move Down!")
                p.position.x = -0.258
                p.position.y = -0.53
                p.position.z = 0.075
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=-15 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 50.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #8")
                time.sleep(1)

                # Move 9) Move straight to Rail
                print("Phase #9: Move straight to Rail!")
                p.position.x = -0.258
                p.position.y = -0.505
                p.position.z = 0.074
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=-15 / 180.0 * 3.14159265)
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
                    time.sleep(1.5)


                # Move 11) Regrasp Upwards
                print("Phase #11: Regrasp Upwards")
                p.position.x = -0.258
                p.position.y = -0.53
                p.position.z = 0.1
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 20.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #11")

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

                # Move 12) Position Finger
                print("Phase #12: Position Finger")
                p.position.x = -0.245
                p.position.y = -0.505
                p.position.z = 0.1
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 40.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #12")

                time.sleep(1)

                rospy.wait_for_service('/css/updateContactSituationBlocking_srv')
                try:
                    switchCS = rospy.ServiceProxy('/css/updateContactSituationBlocking_srv', ContactSituation)
                    c = ContactSituationRequest()
                    c.kp_trans.x = 1600.0
                    c.kp_trans.y = 1600.0
                    c.kp_trans.z = 1600.0
                    c.kp_rot.x = 300.0
                    c.kp_rot.y = 300.0
                    c.kp_rot.z = 100.0

                    c.kd_trans.x = 80.0
                    c.kd_trans.y = 80.0
                    c.kd_trans.z = 80.0
                    c.kd_rot.x = 1.2
                    c.kd_rot.y = 1.2
                    c.kd_rot.z = 1.0

                    c.fdir_trans.x = 0.0
                    c.fdir_trans.y = 0.0
                    c.fdir_trans.z = 1.0
                    c.fdir_rot.x = 0.0
                    c.fdir_rot.y = 0.0
                    c.fdir_rot.z = 0.0

                    c.force_trans.x = 0.0
                    c.force_trans.y = 0.0
                    c.force_trans.z = -37.0
                    c.force_rot.x = 0.0
                    c.force_rot.y = 0.0
                    c.force_rot.z = 0.0

                    c.time = 3.0

                    c.update_pose_first = False

                    switchCS(c)
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)

                # rospy.wait_for_service('/css/setFFVec_srv')
                # try:
                #     applyContactForce = rospy.ServiceProxy('/css/setFFVec_srv', ContactForce)
                #     w = Wrench()
                #     w.force.x = 0.0
                #     w.force.y = 5.0
                #     w.force.z = -40.0
                #     w.torque.x = 0.0
                #     w.torque.y = 0.0
                #     w.torque.z = 0.0

                #     applyContactForce(w)
                # except rospy.ServiceException as e:
                #     print("Service call failed: %s"%e)

                time.sleep(1)

                rospy.wait_for_service('/css/updateContactSituationBlocking_srv')
                try:
                    switchCS = rospy.ServiceProxy('/css/updateContactSituationBlocking_srv', ContactSituation)
                    c = ContactSituationRequest()
                    c.kp_trans.x = 1600.0
                    c.kp_trans.y = 1600.0
                    c.kp_trans.z = 1600.0
                    c.kp_rot.x = 300.0
                    c.kp_rot.y = 300.0
                    c.kp_rot.z = 100.0

                    c.kd_trans.x = 80.0
                    c.kd_trans.y = 80.0
                    c.kd_trans.z = 80.0
                    c.kd_rot.x = 1.2
                    c.kd_rot.y = 1.2
                    c.kd_rot.z = 1.0

                    c.fdir_trans.x = 0.0
                    c.fdir_trans.y = 0.0
                    c.fdir_trans.z = 0.0
                    c.fdir_rot.x = 0.0
                    c.fdir_rot.y = 0.0
                    c.fdir_rot.z = 0.0

                    c.force_trans.x = 0.0
                    c.force_trans.y = 0.0
                    c.force_trans.z = 0.0
                    c.force_rot.x = 0.0
                    c.force_rot.y = 0.0
                    c.force_rot.z = 0.0

                    c.time = 1.0

                    c.update_pose_first = True

                    switchCS(c)
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)

                # Move 13) Up Rotate side
                side_deg = -25.0
                print("Phase #13: Up Rotate side")
                p.position.x = -0.245
                p.position.y = -0.505
                p.position.z = 0.1
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, 1, 0], angle=side_deg / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 50.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #13")

                # Move 14) LEFT
                print("Phase #14: LEFT")
                p.position.x = -0.29
                p.position.y = -0.535
                p.position.z = 0.1
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, 1, 0], angle=side_deg / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 20.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #14")

                # Move 15) DOWN
                print("Phase #15: DOWN")
                p.position.x = -0.29
                p.position.y = -0.535
                p.position.z = 0.04
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, 1, 0], angle=side_deg / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 50.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #15")

                rospy.wait_for_service('/css/updateContactSituationBlocking_srv')
                try:
                    switchCS = rospy.ServiceProxy('/css/updateContactSituationBlocking_srv', ContactSituation)
                    c = ContactSituationRequest()
                    c.kp_trans.x = 1600.0
                    c.kp_trans.y = 1600.0
                    c.kp_trans.z = 1600.0
                    c.kp_rot.x = 300.0
                    c.kp_rot.y = 300.0
                    c.kp_rot.z = 100.0

                    c.kd_trans.x = 80.0
                    c.kd_trans.y = 80.0
                    c.kd_trans.z = 80.0
                    c.kd_rot.x = 1.2
                    c.kd_rot.y = 1.2
                    c.kd_rot.z = 1.0

                    c.fdir_trans.x = 1.0
                    c.fdir_trans.y = 0.0
                    c.fdir_trans.z = 0.0
                    c.fdir_rot.x = 0.0
                    c.fdir_rot.y = 0.0
                    c.fdir_rot.z = 0.0

                    c.force_trans.x = 30.0
                    c.force_trans.y = 0.0
                    c.force_trans.z = 0.0
                    c.force_rot.x = 0.0
                    c.force_rot.y = 0.0
                    c.force_rot.z = 0.0

                    c.time = 4.0

                    c.update_pose_first = False

                    switchCS(c)
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)

                # rospy.wait_for_service('/css/setFFVec_srv')
                # try:
                #     applyContactForce = rospy.ServiceProxy('/css/setFFVec_srv', ContactForce)
                #     w = Wrench()
                #     w.force.x = 0.0
                #     w.force.y = 5.0
                #     w.force.z = -40.0
                #     w.torque.x = 0.0
                #     w.torque.y = 0.0
                #     w.torque.z = 0.0

                #     applyContactForce(w)
                # except rospy.ServiceException as e:
                #     print("Service call failed: %s"%e)

                time.sleep(1.5)

                rospy.wait_for_service('/css/updateContactSituationBlocking_srv')
                try:
                    switchCS = rospy.ServiceProxy('/css/updateContactSituationBlocking_srv', ContactSituation)
                    c = ContactSituationRequest()
                    c.kp_trans.x = 1600.0
                    c.kp_trans.y = 1600.0
                    c.kp_trans.z = 1600.0
                    c.kp_rot.x = 300.0
                    c.kp_rot.y = 300.0
                    c.kp_rot.z = 100.0

                    c.kd_trans.x = 80.0
                    c.kd_trans.y = 80.0
                    c.kd_trans.z = 80.0
                    c.kd_rot.x = 1.2
                    c.kd_rot.y = 1.2
                    c.kd_rot.z = 1.0

                    c.fdir_trans.x = 0.0
                    c.fdir_trans.y = 0.0
                    c.fdir_trans.z = 0.0
                    c.fdir_rot.x = 0.0
                    c.fdir_rot.y = 0.0
                    c.fdir_rot.z = 0.0

                    c.force_trans.x = 0.0
                    c.force_trans.y = 0.0
                    c.force_trans.z = 0.0
                    c.force_rot.x = 0.0
                    c.force_rot.y = 0.0
                    c.force_rot.z = 0.0

                    c.time = 1.0

                    c.update_pose_first = True

                    switchCS(c)
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)

                #######################

                # Move 16) UP FINALLY
                print("Phase #16: UP FINALLY")
                p.position.x = -0.29
                p.position.y = -0.535
                p.position.z = 0.2
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 30.0
                m.i_max_rot_sec = 30.0
                resp1 = add_two_ints(m)
                print("Done with Phase #16")
         
                # 3)
                # p.position.x = -0.245
                # p.position.y = -0.505
                # p.position.z = 0.12
                # 4)
                # p.position.x = -0.29
                # p.position.y = -0.535
                # p.position.z = 0.12
                # 5)
                # p.position.x = -0.29
                # p.position.y = -0.535
                # p.position.z = 0.04
                # self.outQuat = self.clamp1Quat * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265) pyquaternion.Quaternion(axis=[1, 1, 0], angle=30.0 / 180.0 * 3.14159265)
                # 6)
                # p.position.x = -0.245
                # p.position.y = -0.535
                # p.position.z = 0.04
                # self.outQuat = self.clamp1Quat * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265) pyquaternion.Quaternion(axis=[1, 1, 0], angle=30.0 / 180.0 * 3.14159265)
                # 7)
                # p.position.x = -0.4
                # p.position.y = -0.535
                # p.position.z = 0.15

                

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

    def listener_ft_wrench(self, data):
        self.lock_wrench.acquire()
        self.wrench_data = data
        self.lock_wrench.release()

    # def moveGuarded(self, cart_direction=[0,0,0], step_width_in_m=0.00001, max_force_in_n=40.0):
    #     # Quaternion(w, x, y, z)
    #     dire = np.array(cart_direction)
    #     step = dire * step_width_in_m
    #     # If no wrench reading return
    #     self.lock_wrench.acquire()
    #     self.wrench_data_internal = self.wrench_data
    #     self.lock_wrench.release()
    #     if self.wrench_data_internal == None:
    #         print("Returning, no wrench reading!")
    #         return
        
    #     force = np.array([self.wrench_data_internal.force.x, self.wrench_data_internal.force.y, self.wrench_data_internal.force.z])
    #     force_length = np.linalg.norm(force * dire)

    #     print("Start Guarded Move")
    #     # add time-based smoothing?
    #     while force_length < max_force_in_n:
    #         self.cur = self.cur + step
    #         self.ros_t.translation.x = self.cur[0]
    #         self.ros_t.translation.y = self.cur[1]
    #         self.ros_t.translation.z = self.cur[2]
    #         self.pub_traj.publish(self.cart_traj_point)
    #         self.rate.sleep()

    #         self.lock_wrench.acquire()
    #         self.wrench_data_internal = self.wrench_data
    #         self.lock_wrench.release()
    #         force = np.array([self.wrench_data_internal.force.x, self.wrench_data_internal.force.y, self.wrench_data_internal.force.z])
    #         force_length = np.linalg.norm(force * dire)

    #     print("Force Sensed!")

if __name__ == "__main__":
    c = ClampIt()