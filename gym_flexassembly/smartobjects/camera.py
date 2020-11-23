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

                        if self._new_data_available:
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

            self._start_time = perf_counter()

            carpos = None
            carorn = None
            if self._link_id:
                linkWorldPosition, linkWorldOrientation, localInertialFramePosition, localInertialFrameOrientation, carpos, carorn = self._p.getLinkState(self._model_id, self._link_id)
            else:
                carpos, carorn = self._p.getBasePositionAndOrientation(self._model_id)
            
            

            # get a new camera image
            


            # invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
            ballPosInCar2, ballOrnInCar2 = self._p.multiplyTransforms(carpos, carorn, [0,0,-0.1], [0,0,0,1])

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

            # print("Pos " + str(carpos))
            # print("Orn " + str(carorn))

            carmat = self._p.getMatrixFromQuaternion(linkWorldOrientation)
            up = [carmat[2], carmat[5], carmat[8]]

            # self._view_matrix = self._p.computeViewMatrix(carpos,
            #                                             ballPosInCar2,
            #                                             up)

            # print("Tos " + str(np.array(carpos) - np.array(self._settings['target_dist'])))

            carrpy = self._p.getEulerFromQuaternion(carorn)
            self._view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                                                    cameraTargetPosition=ballPosInCar2,
                                                    distance=0.001,
                                                    yaw=carrpy[2],
                                                    pitch=carrpy[1],
                                                    roll=carrpy[0],
                                                    upAxisIndex=1)

            # self._view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            #                                         cameraTargetPosition=np.array(carpos) - np.array(self._settings['target_dist']),
            #                                         distance=-0.05,
            #                                         yaw=0,
            #                                         pitch=0,
            #                                         roll=0,
            #                                         upAxisIndex=1)

            # print("Tos " + str(np.array(carpos) - np.array(self._settings['target_dist'])))

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
            self._lock.release()

            self._new_data_available = True

    # def getExtendedObservation(self):
    #     carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    #     carmat = self._p.getMatrixFromQuaternion(carorn)
    #     ballpos, ballorn = self._p.getBasePositionAndOrientation(self._ballUniqueId)
    #     invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
    #     ballPosInCar, ballOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, ballpos, ballorn)
    #     dist0 = 0.3
    #     dist1 = 1.
    #     eyePos = [
    #         carpos[0] + dist0 * carmat[0], carpos[1] + dist0 * carmat[3],
    #         carpos[2] + dist0 * carmat[6] + 0.3
    #     ]
    #     targetPos = [
    #         carpos[0] + dist1 * carmat[0], carpos[1] + dist1 * carmat[3],
    #         carpos[2] + dist1 * carmat[6] + 0.3
    #     ]
    #     up = [carmat[2], carmat[5], carmat[8]]
    #     viewMat = self._p.computeViewMatrix(eyePos, targetPos, up)
    #     #viewMat = self._p.computeViewMatrixFromYawPitchRoll(carpos,1,0,0,0,2)
    #     #print("projectionMatrix:")
    #     #print(self._p.getDebugVisualizerCamera()[3])
    #     projMatrix = [
    #         0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0,
    #         0.0, 0.0, -0.02000020071864128, 0.0
    #     ]
    #     img_arr = self._p.getCameraImage(width=self._width,
    #                                     height=self._height,
    #                                     viewMatrix=viewMat,
    #                                     projectionMatrix=projMatrix)
    #     rgb = img_arr[2]
    #     np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    #     self._observation = np_img_arr
    #     return self._observation

    # def render(self, mode="rgb_array", close=False):
    #     if mode != "rgb_array":
    #     return np.array([])
    #     base_pos = self.minitaur.GetBasePosition()
    #     view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
    #         cameraTargetPosition=base_pos,
    #         distance=self._cam_dist,
    #         yaw=self._cam_yaw,
    #         pitch=self._cam_pitch,
    #         roll=0,
    #         upAxisIndex=2)
    #     proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
    #                                                                 aspect=float(RENDER_WIDTH) /
    #                                                                 RENDER_HEIGHT,
    #                                                                 nearVal=0.1,
    #                                                                 farVal=100.0)
    #     (_, _, px, _, _) = self._pybullet_client.getCameraImage(
    #         width=RENDER_WIDTH,
    #         height=RENDER_HEIGHT,
    #         renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
    #         viewMatrix=view_matrix,
    #         projectionMatrix=proj_matrix)
    #     rgb_array = np.array(px)
    #     rgb_array = rgb_array[:, :, :3]
    #     return rgb_array

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