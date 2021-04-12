import os, inspect

import pybullet as p
import numpy as np
import copy
import math

import sys
import threading

from time import perf_counter

class Camera:
    def __init__(self, pybullet, settings, name, model_id, link_id=None, use_real_interface=True):
        self._settings = settings
        self._name = name
        self._model_id = model_id
        self._link_id = link_id
        self._use_real_interface = use_real_interface
        self._terminate_thread = False
        self._p = pybullet

        self._depth = []
        self._rgb = []

        self._lock = threading.Lock()

        self._new_data_available = False

        self._thread = None

        # self.setPose(self._settings['pos'], self._settings['orn'])

        self._start_time = perf_counter()

        self._wait_frame_rate = 1.0 / self._settings['framerate']

        self._near = self._settings['near']
        self._far = self._settings['far']

        self.dl0 = None
        self.dl1 = None
        self.dl2 = None

        if self._use_real_interface:
            # load (optional) ROS imports if the real interface should be mirrored
            try:
                import rospy
                from std_msgs.msg import Header
                from geometry_msgs.msg import Pose
                from sensor_msgs.msg import Image
                import cv_bridge

                # not sure if we really need this
                global rospy, cv_bridge, Pose, Image, Header

                # print(self._p.isNumpyEnabled())

                # create ros publisher
                self._pub_depth = rospy.Publisher('~camera/' + self._name + '/depth/image_raw', Image, queue_size=1)
                self._pub_color = rospy.Publisher('~camera/' + self._name + '/color/image_raw', Image, queue_size=1)
                # instantiate the cv bridge
                self._bridge = cv_bridge.CvBridge()

                self._rate = rospy.Rate(self._settings['framerate'])

                print("\n\t> Initialized camera " + str(self._name) + " depth image \n\t(sensor_msgs.Image) on ~camera/"+str(self._name)+"/depth/image_raw publisher\n")

                print("\n\t> Initialized camera " + str(self._name) + " color image \n\t(sensor_msgs.Image) on ~camera/"+str(self._name)+"/color/image_raw publisher\n")

                def thread_func():
                    while not self._terminate_thread:
                        save = False
                        self._lock.acquire()
                        save = self._new_data_available
                        self._lock.release()

                        if save:
                            # create a header for the messages
                            header = Header()
                            header.stamp = rospy.Time.now()

                            self._lock.acquire()
                            try:
                                img_depth = self._bridge.cv2_to_imgmsg(self._depth, 'passthrough')
                                img_color = self._bridge.cv2_to_imgmsg(self._rgb, 'passthrough')
                            except cv_bridge.CvBridgeError as e:
                                print('Could not convert CV image to ROS msg: ' + str(e))
                            self._lock.release()

                            # publish the images
                            img_depth.header = header
                            img_depth.header.frame_id = 'depth_image'
                            self._pub_depth.publish(img_depth)

                            img_color.header = header
                            img_color.header.frame_id = 'color_image'
                            self._pub_color.publish(img_color)

                            self._new_data_available = False

                        self._rate.sleep()

                # Start the ROS publishing thread
                self._thread = threading.Thread(target=thread_func, daemon=True)
                self._thread.start()
            
            except ImportError:
                print("ERROR IMPORTING ros camera", file=sys.stderr)
        
        self.reset()

    def update(self):
        if not self._terminate_thread:
            if (perf_counter() - self._start_time) < self._wait_frame_rate:
                return

            if self._lock.locked():
                return

            self._start_time = perf_counter()

            carpos = None
            carorn = None
            if self._link_id:
                linkWorldPosition, linkWorldOrientation, localInertialFramePosition, localInertialFrameOrientation, carpos, carorn = self._p.getLinkState(self._model_id, self._link_id)
            else:
                carpos, carorn = self._p.getBasePositionAndOrientation(self._model_id)
            
            # get a new camera image

            # invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
            ballPosInCar2, ballOrnInCar2 = self._p.multiplyTransforms(carpos, carorn, [0,0,0.1], [0,0,0,1])

            # targetInWTrans, targetInWOrn = self._p.multiplyTransforms(invCarPos, invCarOrn, ballPosInCar2, ballOrnInCar2)

            if self.dl0:
                self.dl0 = self._p.addUserDebugLine([0,0,0], [0,0,0.05], [1,0,0], lineWidth=4, parentObjectUniqueId=self._model_id, parentLinkIndex=self._link_id, replaceItemUniqueId=self.dl0)
            else:
                self.dl0 = self._p.addUserDebugLine([0,0,0], [0,0,0.05], [1,0,0], lineWidth=4, parentObjectUniqueId=self._model_id, parentLinkIndex=self._link_id)

            if self.dl1:
                self.dl1 = self._p.addUserDebugLine(carpos, ballPosInCar2, [0,1,0], lineWidth=5, replaceItemUniqueId=self.dl1)
            else:
                self.dl1 = self._p.addUserDebugLine(carpos, ballPosInCar2, [0,1,0], lineWidth=5)

            # if self.dl2:
            #     self.dl2 = self._p.addUserDebugLine(carpos, targetInWTrans, [0,0,1], lineWidth=5, replaceItemUniqueId=self.dl2)
            # else:
            #     self.dl2 = self._p.addUserDebugLine(carpos, targetInWTrans, [0,0,1], lineWidth=5)

            # carmat = self._p.getMatrixFromQuaternion(ballOrnInCar2)
            # up = [carmat[2], carmat[5], carmat[8]]

            t1, t2 = self._p.multiplyTransforms([0,0,0], ballOrnInCar2, [0,0,0], self._p.getQuaternionFromEuler([0,-1.5708,0]))
            carrpy2 = self._p.getEulerFromQuaternion(t2)
            # model_transformation_orn = np.array([3.14159, 0, 1.5708])

            self._view_matrix = self._p.computeViewMatrix(carpos,
                                                        ballPosInCar2,
                                                        carrpy2)

            # carrpy = self._p.getEulerFromQuaternion(carorn)
            # self._view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            #                                         cameraTargetPosition=ballPosInCar2,
            #                                         distance=-0.001,
            #                                         yaw=carrpy[2],
            #                                         pitch=carrpy[1],
            #                                         roll=carrpy[0],
            #                                         upAxisIndex=1)

            # self._view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            #                                         cameraTargetPosition=np.array(carpos) - np.array(self._settings['target_dist']),
            #                                         distance=-0.05,
            #                                         yaw=0,
            #                                         pitch=0,
            #                                         roll=0,
            #                                         upAxisIndex=1)

            # self._settings['up']
            self._projection_matrix = self._p.computeProjectionMatrixFOV(self._settings['fov'],
                                                                        self._settings['width'] / self._settings['height'],
                                                                        self._settings['near'],
                                                                        self._settings['far'])

            _, _, rgba, depth_buffer, _ = self._p.getCameraImage(self._settings['width'],
                                                                self._settings['height'],
                                                                self._view_matrix,
                                                                self._projection_matrix,
                                                                shadow=True,
                                                                renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
            self._lock.acquire()
            self._depth = self._far * self._near / (self._far - (self._far - self._near) * depth_buffer)
            self._rgb = rgba[:, :, :3]
            self._new_data_available = True
            self._lock.release()

    def getModelId(self):
        return self._model_id

    def setPose(self, pos, orn):
        self._settings['pos'] = pos
        self._settings['orn'] = orn
        p.resetBasePositionAndOrientation(self._model_id, self._settings['pos'], self._settings['orn'])

    def getPose(self):
        # TODO this does not support moving cameras yet
        return self._settings['pos'], self._settings['orn']

    def reset(self):
        # Find indices associated with links names
        # total_joints = p.getNumJoints(self._model_id)
        # for i in range(total_joints):
            # jointInfo = p.getJointInfo(self._model_id, i)
            # print("?????????????? NAME: " + str(jointInfo[1].decode('UTF-8')) + " " + str(i))
            # if str("iiwa7_joint_camera_mount_realsense") == (str(jointInfo[1].decode('UTF-8'))):
            # self._finger_1_joint_index = i
            # pass
        pass

    def getUUid(self):
        return self._model_id

    def getObservation(self):
        observation = []
        return observation

    def terminate(self):
        self._terminate_thread = True
        self._thread.join()

        self._pub_color.unregister()
        self._pub_depth.unregister()

    def __del__(self):
        self.terminate()

__all__ = ['Camera']