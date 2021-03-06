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
        self.clamp1.position.x = -0.348
        self.clamp1.position.y = -0.63

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
            rospy.wait_for_service('/css/move_async_srv')
            rospy.wait_for_service('pose_estimation')
            time.sleep(1)
            try:
                move_service = rospy.ServiceProxy('/css/move_srv', Move)
                move_async_service = rospy.ServiceProxy('/css/move_async_srv', Move)

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
                m.i_max_trans_sec = 10.0
                m.i_max_rot_sec = 20.0
                resp1 = move_service(m)
                print("Done with Phase #1")

                time.sleep(2)

                # Move 2) Rotate to initial observer pose
                print("Phase #2: Rotate to initial observer pose!")
                p.position.x = -0.0452333
                p.position.y = -0.594722
                p.position.z = 0.301398
                self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=0.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 8.0
                m.i_max_rot_sec = 10.0
                resp1 = move_service(m)
                print("Done with Phase #2")
                time.sleep(2)

                # Move 3) Move to observer pose
                print("Phase #3: Move to observer pose!")
                # p.position.x = -0.25
                # p.position.y = -0.594722
                # p.position.z = 0.45
                p.position.x = -0.25
                p.position.y = -0.594722
                p.position.z = 0.35
                self.outQuat = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=0.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 8.0
                m.i_max_rot_sec = 20.0
                resp1 = move_service(m)
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
                #  * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 5.0
                m.i_max_rot_sec = 8.0
                resp1 = move_service(m)
                print("Done with Phase #4")
                time.sleep(1)

                # Move 5) Move to grasp clamp 1
                print("Phase #5: Move to grasp clamp 1!")
                p.position.x = self.clamp1.position.x
                p.position.y = self.clamp1.position.y
                p.position.z = 0.065
                self.outQuat = self.clamp1Quat
                #  * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 20.0
                m.i_max_rot_sec = 10.0
                resp1 = move_service(m)
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
                p.position.z = 0.12
                self.outQuat = self.clamp1Quat
                #  * pyquaternion.Quaternion(axis=[0, 0, 1], angle=-135.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 20.0
                m.i_max_rot_sec = 30.0
                resp1 = move_service(m)
                print("Done with Phase #6")
                time.sleep(1)

                # Move 7) Move to rail drag pose
                print("Phase #7: Move to rail drag pose!")
                p.position.x = -0.258
                p.position.y = -0.57
                p.position.z = 0.1
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=-15 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 20.0
                m.i_max_rot_sec = 30.0
                resp1 = move_service(m)
                print("Done with Phase #7")
                time.sleep(1)

                # Move 8) Move Down
                print("Phase #8: Move Down!")
                p.position.x = -0.258
                p.position.y = -0.57
                p.position.z = 0.0
                # p.position.z = 0.0
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=-15 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 50.0
                m.i_max_rot_sec = 30.0
                resp1 = move_async_service(m)
                print("Done with Phase #8")

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
                    c.force_trans.z = -4.0
                    c.force_rot.x = 0.0
                    c.force_rot.y = 0.0
                    c.force_rot.z = 0.0

                    c.time = 0.1

                    c.update_pose_first = False

                    self.lock_wrench.acquire()
                    self.wrench_data_internal = self.wrench_data
                    self.lock_wrench.release()
                    if self.wrench_data_internal == None:
                        print("Sleeping for 5 secs. Fallback: no wrench reading!")
                        time.sleep(3)
                    else:
                        since = time.time()
                        while (time.time() - since) < 7.0:
                            self.lock_wrench.acquire()
                            self.wrench_data_internal = self.wrench_data
                            self.lock_wrench.release()
                            if self.wrench_data_internal.force.z > 6.0:
                                break
                            time.sleep(0.2)

                    switchCS(c)
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)

                # Move 9) Move straight to Rail

                rospy.wait_for_service('/css/updateContactSituationBlocking_srv')
                print("Phase #9: Move straight to Rail!")
                p.position.x = -0.258
                p.position.y = -0.5
                p.position.z = 0.0
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265) * pyquaternion.Quaternion(axis=[1, -1, 0], angle=-15 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 50.0
                m.i_max_rot_sec = 30.0
                resp1 = move_async_service(m)
                print("Done with Phase #9")
                # time.sleep(1)

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
                    c.fdir_trans.y = 1.0
                    c.fdir_trans.z = 1.0
                    c.fdir_rot.x = 1.0
                    c.fdir_rot.y = 0.0
                    c.fdir_rot.z = 0.0

                    c.force_trans.x = 0.0
                    c.force_trans.y = 10.0
                    c.force_trans.z = 0.0
                    c.force_rot.x = 0.0
                    c.force_rot.y = 0.0
                    c.force_rot.z = 0.0

                    c.time = 1.5

                    c.update_pose_first = False

                    self.lock_wrench.acquire()
                    self.wrench_data_internal = self.wrench_data
                    self.lock_wrench.release()
                    if self.wrench_data_internal == None:
                        print("Sleeping for 5 secs. Fallback: no wrench reading!")
                        time.sleep(3)
                    else:
                        since = time.time()
                        while (time.time() - since) < 20.0:
                            self.lock_wrench.acquire()
                            self.wrench_data_internal = self.wrench_data
                            self.lock_wrench.release()
                            if self.wrench_data_internal.force.y < -6.0:
                                print("Y reached!")
                                break
                            time.sleep(0.2)
                        print("Escaped!")

                    switchCS(c)
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)

                time.sleep(1)

                rospy.wait_for_service('/css/setFFVec_srv')
                try:
                    applyContactForce = rospy.ServiceProxy('/css/setFFVec_srv', ContactForce)
                    w = Wrench()
                    w.force.x = 0.0
                    w.force.y = 10.0
                    w.force.z = -35.0
                    w.torque.x = 0.0
                    w.torque.y = 0.0
                    w.torque.z = 0.0

                    applyContactForce(w)
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)

                self.lock_wrench.acquire()
                self.wrench_data_internal = self.wrench_data
                self.lock_wrench.release()
                if self.wrench_data_internal == None:
                    print("Sleeping for 5 secs. Fallback: no wrench reading!")
                    time.sleep(5)
                else:
                    since = time.time()
                    while (time.time() - since) < 5.0:
                        self.lock_wrench.acquire()
                        self.wrench_data_internal = self.wrench_data
                        self.lock_wrench.release()
                        if self.wrench_data_internal.force.z > 38.0:
                            break
                        time.sleep(0.2)

                # time.sleep(0.5)

                

                rospy.wait_for_service('/css/setFFVec_srv')
                try:
                    applyContactForce = rospy.ServiceProxy('/css/setFFVec_srv', ContactForce)
                    w = Wrench()
                    w.force.x = 0.0
                    w.force.y = 0.0
                    w.force.z = 0.0
                    w.torque.x = 0.0
                    w.torque.y = 0.0
                    w.torque.z = 0.0

                    applyContactForce(w)
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)

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

                    c.time = 3.0

                    c.update_pose_first = True

                    switchCS(c)
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)

                # time.sleep(1)

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

                return

                # Move 11) Regrasp
                print("Phase #11: Regrasp")
                p.position.x = -0.258
                p.position.y = -0.53
                p.position.z = 0.049
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 50.0
                m.i_max_rot_sec = 20.0
                resp1 = move_service(m)
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

                # Move 12) Slide
                print("Phase #12: Slide")
                p.position.x = -0.21
                p.position.y = -0.53
                p.position.z = 0.049
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 50.0
                m.i_max_rot_sec = 30.0
                resp1 = move_service(m)
                print("Done with Phase #12")

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
                    time.sleep(1)


                # Move 13) Up Again
                print("Phase #13: Up Again")
                p.position.x = -0.21
                p.position.y = -0.53
                p.position.z = 0.15
                self.outQuat = pyquaternion.Quaternion(w=0,x=0,y=1,z=0) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=45.0 / 180.0 * 3.14159265)
                p.orientation.x = self.outQuat[1]
                p.orientation.y = self.outQuat[2]
                p.orientation.z = self.outQuat[3]
                p.orientation.w = self.outQuat[0]
                m.i_pose = p
                m.i_max_trans_sec = 10.0
                m.i_max_rot_sec = 30.0
                resp1 = move_service(m)
                print("Done with Phase #13")

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

if __name__ == "__main__":
    c = ClampIt()