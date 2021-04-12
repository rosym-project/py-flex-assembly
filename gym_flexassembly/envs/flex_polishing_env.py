#!/usr/bin/python3

""" MAIN: Environment for the window polishing scenario.
:Author:
  `Dennis Leroy Wigand <dwigand@cor-lab.de>`
"""

# SYSTEM IMPORTS
import os, inspect

# UTILITY IMPORTS
import math
import numpy as np
import random
import time
import sys

# PYBULLET IMPORTS
import pybullet as p
import pybullet_data

# GYM IMPORTS
import gym
from gym import error, spaces, utils
from gym.utils import seeding


# DEPLOYMENT IMPORTS
from pkg_resources import parse_version

# FLEX ASSEMBLY DATA IMPORTS
from gym_flexassembly import data as flexassembly_data

# FLEX ASSEMBLY ROBOT IMPORTS
from gym_flexassembly.robots.kuka_iiwa import KukaIIWA, KukaIIWA7, KukaIIWA14
from gym_flexassembly.robots.kuka_iiwa_egp_40 import KukaIIWA_EGP40, KukaIIWA7_EGP40
from gym_flexassembly.robots.prismatic_2_finger_gripper_plugin import Prismatic2FingerGripperPlugin

# FLEX ASSEMBLY SMARTOBJECTS IMPORTS
from gym_flexassembly.smartobjects.spring_clamp import SpringClamp

from gym_flexassembly.envs.env_interface import EnvInterface

from gym_flexassembly.constraints import frame

class FlexPolishingEnv(EnvInterface):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50} # TODO do we need this?)

    def __init__(self,
               stepping=True,
               gui=True,
               direct=False,
               use_real_interface=True,
               static=False):
        super().__init__(gui, direct, use_real_interface=use_real_interface, hz=1000.0, stepping=stepping)

        self._urdfRoot_pybullet = pybullet_data.getDataPath()
        self._urdfRoot_flexassembly = flexassembly_data.getDataPath()
        # self._observation = []
        # self._cam_dist = 1.3
        # self._cam_yaw = 180
        # self._cam_pitch = -40

        self.largeValObservation = 100 # TODO
        self.RENDER_HEIGHT = 720 # TODO
        self.RENDER_WIDTH = 960 # TODO

        self.me_line_1 = p.addUserDebugLine([0,0,0], [0,0,0], [0.9, 0.6, 0.0], 5)
        self.me_line_1_a = p.addUserDebugLine([0,0,0], [0,0,0], [0.9, 0.6, 0.0], 5)
        self.me_line_1_b = p.addUserDebugLine([0,0,0], [0,0,0], [0.9, 0.6, 0.0], 5)
        self.me_line_2 = p.addUserDebugLine([0,0,0], [0,0,0], [0.9, 0.6, 0.0], 5)

        self.seed()

        self.collect_contact = []

        self.cam_global_settings = {'width': 1280,
                                    'height': 720,
                                    'fov': 65,
                                    'near': 0.16,
                                    'far': 10,
                                    'framerate': 5,
                                    'up': [0, -1.0, 0]}
        self.object_ids = {}

        self.env_reset()

        self.startTime = time.time_ns()

        self.itercount = 0
        self.gogogo = True

        # self.env_loop() # TODO

    def loadEnvironment(self):
        # print("pybullet_data.getDataPath() = " + str(pybullet_data.getDataPath()))
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # plate_id = self._p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/plate.urdf", useFixedBase=True)
        # self._p.resetBasePositionAndOrientation(plate_id, [0,0,0.3], [0, 0, 0, 1])

        # Disable rendering
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

        # # Floor SHOULD BE ALWAYS ID 0
        # self._p.loadURDF(os.path.join(self._urdfRoot_flexassembly, "objects/plane_solid.urdf"), useMaximalCoordinates=True) # Brauche ich fuer die hit rays

        self.window_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/objects/window.urdf", useFixedBase=True)
        self.window_pos = np.array([0.72,0.13,1.22])
        p.resetBasePositionAndOrientation(self.window_id, self.window_pos, [0,0,0,1])
        self.object_ids['window'] = self.window_id

        ##########################
        ###   Draw Traj Path   ###
        ##########################

        start = np.array([0.71,0.25,1.7])
        self.robot_start = start
        end = start + np.array([0,-0.25,0])
        # p.addUserDebugLine(start, end, [0.3, 0.3, 0.3], 2)
        ccolor = [0.3, 0.3, 0.3]
        # dir
        # p.addUserDebugLine(start+0.5*(end-start), start+0.5*(end-start)+[0,0.03,0.03], [0.3, 0.3, 0.3], 2)
        # p.addUserDebugLine(start+0.5*(end-start), start+0.5*(end-start)+[0,0.03,-0.03], [0.3, 0.3, 0.3], 2)

        # for i in range(0,7):
        #     start = end
        #     end = end + np.array([0,0,-0.05])
        #     p.addUserDebugLine(start, end, [0.3, 0.3, 0.3], 2)

        #     start = end
        #     if i % 2:
        #         end = end + np.array([0,-0.25,0])
        #         dir1 = [0,0.03,0.03]
        #         dir2 = [0,0.03,-0.03]
        #     else:
        #         end = end + np.array([0,0.25,0])
        #         dir1 = [0,-0.03,0.03]
        #         dir2 = [0,-0.03,-0.03]
        #     p.addUserDebugLine(start, end, [0.3, 0.3, 0.3], 2)
        #     # dir
        #     p.addUserDebugLine(start+0.5*(end-start), start+0.5*(end-start)+dir1, [0.3, 0.3, 0.3], 2)
        #     p.addUserDebugLine(start+0.5*(end-start), start+0.5*(end-start)+dir2, [0.3, 0.3, 0.3], 2)


        # self.addVisFrame([0.4, 0.2, 0.8],'O1')

        # Enable rendering again
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    def loadRobot(self):
        # Disable rendering
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
        # os.path.join(flexassembly_data.getDataPath(), "robots/epfl-iiwa14/iiwa14.urdf")
        # Load robot KUKA IIWA 14
        # self.kuka14_1 = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/epfl-iiwa14/iiwa14.urdf", useFixedBase=True)
        # if self._use_real_interface:
        #     f = open("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/epfl-iiwa14/iiwa14.urdf","r") # TODO
        #     self.upload_urdf(f.read(), "robot_description")

        self.kuka14_1 = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/epfl-iiwa14-wischer/iiwa14.urdf", useFixedBase=True)
        if self._use_real_interface:
            f = open("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/epfl-iiwa14-wischer/iiwa14.urdf","r") # TODO
            self.upload_urdf(f.read(), "robot_description")
        
        self._p.resetBasePositionAndOrientation(self.kuka14_1, [0,0,0.7], [0,0,0,1])
        # TODO
        # self._p.resetJointState(self.kuka14_1, 1, 0.0, 0.0)
        # self._p.resetJointState(self.kuka14_1, 2, 0.0, 0.0)
        # self._p.resetJointState(self.kuka14_1, 3, 0.0, 0.0)
        # self._p.resetJointState(self.kuka14_1, 4, -0.884, 0.0)
        # self._p.resetJointState(self.kuka14_1, 5, 0.0, 0.0)
        # self._p.resetJointState(self.kuka14_1, 6, 0.632, 0.0)
        # self._p.resetJointState(self.kuka14_1, 7, 0.0, 0.0)

        self._p.resetJointState(self.kuka14_1, 1, 0.196338, 0.0)
        self._p.resetJointState(self.kuka14_1, 2, -0.00784231, 0.0)
        self._p.resetJointState(self.kuka14_1, 3, 0.212446, 0.0)
        self._p.resetJointState(self.kuka14_1, 4, -1.47407, 0.0)
        self._p.resetJointState(self.kuka14_1, 5, -1.33124, 0.0)
        self._p.resetJointState(self.kuka14_1, 6, 0.422643, 0.0)
        self._p.resetJointState(self.kuka14_1, 7, 2.10829, 0.0)

        self._p.resetJointState(self.kuka14_1, 12, 0.01, 0.0)
        self._p.resetJointState(self.kuka14_1, 13, 0.01, 0.0)

        


        # Enable rendering again
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        # Store name with as unique identified + "_0" and the id
        self._robot_map[str(self._p.getBodyInfo(self.kuka14_1)[1].decode()) + "_0"] = self.kuka14_1

        # EEF camera
        self.cam_global_settings['pos'] = [0,0,0]
        self.cam_global_settings['orn'] = [0, 0, 0, 1]
        # self.cam_global_settings['target_dist'] = [0, 0, 0.85]
        self.cam_global_settings['target_dist'] = [0, 0, 0.05]
        self.cam_global_settings['target_pos'] = list(np.array(self.cam_global_settings['pos']) - np.array(self.cam_global_settings['target_dist']))
        self.cam_global_settings['up'] = [-1, 0, 0]
        # tmp_name = str(self._p.getBodyInfo(realsense_camera_id)[1].decode()) + "_0"
        tmp_name = "eefcam_0"
        self._camera_map[tmp_name] = {'model_id':self.kuka14_1, 'link_id':10}

        # Load gripper
        self.kuka7_1_egp = Prismatic2FingerGripperPlugin(self.kuka14_1, "gripper1", "SchunkEGP40_Finger1_joint", "SchunkEGP40_Finger2_joint", use_real_interface=self._use_real_interface)

        kmr_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/kuka-kmr/model.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(kmr_id, [-0.17,-0.36,0], [0,0,0.757,0.757])

        # self._p.addUserDebugLine([-0.1, 0+0.31, 0+0.75], [0.1, 0+0.31, 0+0.75], [1, 0, 0])
        # self._p.addUserDebugLine([0, -0.1+0.31, 0+0.75], [0, 0.1+0.31, 0+0.75], [0, 1, 0])
        # self._p.addUserDebugLine([0, 0+0.31, -0.1+0.75], [0, 0+0.31, 0.1+0.75], [0, 0, 1])

  
        self.link_ft = 9 # 8?
        self._ft_map["ft_0"] = {'model_id':self.kuka14_1, 'link_id':self.link_ft}
        
        # self._p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_ft)
        # self._p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_ft)
        # self._p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_ft)
        # self.frame_text_node = self._p.addUserDebugText(str("Link FT"), [0, 0.15, 0.15],
        #                 textColorRGB=[0, 0, 0],
        #                 textSize=1.0,
        #                 parentObjectUniqueId=self.kuka14_1,
        #                 parentLinkIndex=link_ft)
        # self._p.addUserDebugLine([0, 0.05, 0.05], [0, 0.14, 0.14], [0, 0, 0], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_ft)

        # self.tmp_frame = frame.Frame(self._p, "test", fixed_base=True, ref_id=self.kuka14_1, ref_link_id=9, ref_name="", is_body_frame=True)
        # link_idd = 11
        # self._p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_idd)
        # self._p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_idd)
        # self._p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_idd)
        # self.frame_text_node = self._p.addUserDebugText(str("Link 11"), [0, 0.15, 0.15],
        #                 textColorRGB=[0, 0, 0],
        #                 textSize=1.0,
        #                 parentObjectUniqueId=self.kuka14_1,
        #                 parentLinkIndex=link_idd)
        # self._p.addUserDebugLine([0, 0.05, 0.05], [0, 0.14, 0.14], [0, 0, 0], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_idd)

        # self.a1 = self._p.addUserDebugLine([0,0,0], [0,0,0], [1, 0, 0])
        # self.a2 = self._p.addUserDebugLine([0,0,0], [0,0,0], [0, 1, 0])
        # self.a3 = self._p.addUserDebugLine([0,0,0], [0,0,0], [0, 0, 1])

        # self.addVisFrameAt(kuka14_1,link_idd,'F1')

        # eef_height = 0.1
        # self._p.addUserDebugLine([0, 0, eef_height], [0.1, 0, eef_height], [1, 0, 0], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_idd)
        # self._p.addUserDebugLine([0, 0, eef_height], [0, 0.1, eef_height], [0, 1, 0], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_idd)
        # self._p.addUserDebugLine([0, 0, eef_height], [0, 0, eef_height+0.1], [0, 0, 1], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_idd)
        # self.frame_text_node = self._p.addUserDebugText(str("EEF"), [0, 0.15, eef_height+0.15],
        #                 textColorRGB=[0, 0, 0],
        #                 textSize=1.0,
        #                 parentObjectUniqueId=self.kuka14_1,
        #                 parentLinkIndex=link_idd)
        # self._p.addUserDebugLine([0, 0.05, eef_height+0.05], [0, 0.14, eef_height+0.14], [0, 0, 0], parentObjectUniqueId=self.kuka14_1, parentLinkIndex=link_idd)


    def addVisFrameAt(self,kuka14_1,link_idd,text):
        self._p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=kuka14_1, parentLinkIndex=link_idd)
        self._p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=kuka14_1, parentLinkIndex=link_idd)
        self._p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=kuka14_1, parentLinkIndex=link_idd)
        self.frame_text_node = self._p.addUserDebugText(str(text), [0, 0.15, 0.15],
                        textColorRGB=[0, 0, 0],
                        textSize=1.0,
                        parentObjectUniqueId=kuka14_1,
                        parentLinkIndex=link_idd)
        self._p.addUserDebugLine([0, 0.05, 0.05], [0, 0.14, 0.14], [0, 0, 0], parentObjectUniqueId=kuka14_1, parentLinkIndex=link_idd)


    def addVisFrame(self,pos,text):
        self._p.addUserDebugLine([pos[0], pos[1], pos[2]], [pos[0]+0.1, pos[1]+0, pos[2]+0], [1, 0, 0])
        self._p.addUserDebugLine([pos[0], pos[1], pos[2]], [pos[0]+0, pos[1]+0.1, pos[2]+0], [0, 1, 0])
        self._p.addUserDebugLine([pos[0], pos[1], pos[2]], [pos[0]+0, pos[1]+0, pos[2]+0.1], [0, 0, 1])
        self.frame_text_node = self._p.addUserDebugText(str(text), [pos[0]+0, pos[1]+0.15, pos[2]+0.15],
                        textColorRGB=[0, 0, 0],
                        textSize=1.0)
        self._p.addUserDebugLine([pos[0], pos[1]+0.05, pos[2]+0.05], [pos[0], pos[1]+0.14, pos[2]+0.14], [0, 0, 0])


    def loadCameras(self):
        if not self._use_real_interface:
            return

        for k,v in self._camera_map.items():
            self.remove_camera(name=k)
            self.add_camera(settings=self.cam_global_settings, name=k, model_id=v['model_id'], link_id=v['link_id'])

    def loadFTs(self):
        if not self._use_real_interface:
            return

        for k,v in self._ft_map.items():
            self.remove_ft(name=k)
            self.add_ft(name=k, model_id=v['model_id'], link_id=v['link_id'])

    def step_internal(self):
        if self.kuka7_1_egp:
            self.kuka7_1_egp.update()

        self.getFrameManager().updateFramePoses()

        # pose_frame_tx = self._p.readUserDebugParameter(self.dp_frame_tx)
        # pose_frame_ty = self._p.readUserDebugParameter(self.dp_frame_ty)
        # pose_frame_tz = self._p.readUserDebugParameter(self.dp_frame_tz)
        # pose_frame_rr = self._p.readUserDebugParameter(self.dp_frame_rr)
        # pose_frame_rp = self._p.readUserDebugParameter(self.dp_frame_rp)
        # pose_frame_ry = self._p.readUserDebugParameter(self.dp_frame_ry)

        # cj0 = self._p.readUserDebugParameter(self.dp_j0)
        # cj1 = self._p.readUserDebugParameter(self.dp_j1)
        # cj2 = self._p.readUserDebugParameter(self.dp_j2)
        # cj3 = self._p.readUserDebugParameter(self.dp_j3)
        # cj4 = self._p.readUserDebugParameter(self.dp_j4)
        # cj5 = self._p.readUserDebugParameter(self.dp_j5)
        # cj6 = self._p.readUserDebugParameter(self.dp_j6)
        # ccc = [cj0,cj1,cj2,cj3,cj4,cj5,cj6]

        # # for i in range(1,8):
        # #     self._p.setJointMotorControl2(self.kuka14_1,
        # #                             i,
        # #                             self._p.POSITION_CONTROL,
        # #                             targetPosition=ccc[i-1],
        # #                             force=1000.0)

        # self._p.setJointMotorControl2(self.kuka14_1,
        #                             12,
        #                             self._p.POSITION_CONTROL,
        #                             targetPosition=0,
        #                             force=90000.0)
        # self._p.setJointMotorControl2(self.kuka14_1,
        #                             13,
        #                             self._p.POSITION_CONTROL,
        #                             targetPosition=0,
        #                             force=90000.0)

        if self.itercount > 100 and self.gogogo:
            contacts = self._p.getContactPoints(bodyA=self.kuka14_1, bodyB=self.window_id, linkIndexB=0)
            # contactFlag, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, positionOnA, positionOnB, contactNormalOnB, contactDistance, normalForce, lateralFriction1, lateralFrictionDir1, lateralFriction2, lateralFrictionDir2
            for contact in contacts:
                if contact[3] == 12:
                    # link_index = contact[3]
                    # print(link_index)
                    # ddd = 40000.0
                    # self._p.addUserDebugLine(contact[6], np.array(contact[6]) + np.array(contact[7])*0.001*contact[9]*0.005, [contact[9]/ddd, 0, (ddd-contact[9])/ddd], 4)
                    # print(contact[9])

                    # self.collect_contact.append([contact[6],np.array(contact[6]) + np.array(contact[7])*0.001*contact[9]*0.005])
                    # self.itercount = 0
                    pass
                elif contact[3] == 14:
                    # link_index = contact[3]
                    # print(link_index)
                    ddd = 1000.0
                    # self._p.addUserDebugLine(contact[6], np.array(contact[6]) + np.array(contact[7])*contact[9]*0.001, [contact[9]/ddd, 0, (ddd-contact[9])/ddd], 4)

                    _, _, ft_sensor_forces, _ = self._p.getJointState(self.kuka14_1, 9)
                    # self._p.addUserDebugLine(contact[6], np.array(contact[6]) + np.array(contact[7])*ft_sensor_forces[2]*0.001, [ft_sensor_forces[2]/ddd, 0, (ddd-ft_sensor_forces[2])/ddd], 4)

                    # self.collect_contact.append([contact[6],np.array(contact[6]) + np.array(contact[7])*0.001*contact[9]*0.005])
                    # self.itercount = 0
                    # print(contact[9])
                # if contact[3] == 12 or contact[3] == 13:
                #     # contact[6]
                #     _, _, ft_sensor_forces, _ = self._p.getJointState(self.kuka14_1, 9)
                #     self.collect_contact.append([contact[6],ft_sensor_forces[2]])
                #     self.itercount = 0

        # print(time.time_ns())
        if (time.time_ns() - self.startTime) > 1000:
            self.startTime = time.time_ns()
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if (k == p.B3G_RETURN and (v & p.KEY_WAS_TRIGGERED)):
                    print("Draw")
                    self.gogogo = False
                    # print(self.collect_contact)
                    last = np.array([0,0,0])
                    count = 0
                    print("size: " + str(len(self.collect_contact)))
                    for a in self.collect_contact:
                        atmp = np.array(a[0])
                        if math.fabs(np.linalg.norm(last-atmp)) < 0.1:
                            continue
                        last = atmp
                        self._p.addUserDebugLine(last, last + np.array([-0.001,0,0])*a[1], [1,0,0], 4)
                        count = count + 1
                    print(count)
                if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
                    self.collect_contact = []
                    print("pressed")
        
        self.itercount = self.itercount + 1


        _, _, _, _, worldLinkFramePosition, worldLinkFrameOrientation = self._p.getLinkState(self.kuka14_1, 14)
        self._p.resetBasePositionAndOrientation(self.constraint_edge_glas_id, worldLinkFramePosition, worldLinkFrameOrientation)

        worldLinkFramePosition2, worldLinkFrameOrientation2 = self._p.getBasePositionAndOrientation(self.constraint_glas_id)

        third = (np.array(worldLinkFramePosition)-np.array([0.7,0.0,1.6])) / 3.0

        self.me_line_1 = p.addUserDebugLine(worldLinkFramePosition, np.array(worldLinkFramePosition)-third-np.array([0,0,0.05]), [0.0, 0.6, 0.85], 6, replaceItemUniqueId=self.me_line_1)
        self.me_line_1_a = p.addUserDebugLine(np.array(worldLinkFramePosition)-third-np.array([0,0,0.05]), np.array(worldLinkFramePosition)-third-third-np.array([0,0,-0.05]), [0.0, 0.6, 0.85], 6, replaceItemUniqueId=self.me_line_1_a)
        self.me_line_1_b = p.addUserDebugLine(np.array(worldLinkFramePosition)-third-third-np.array([0,0,-0.05]), [0.7,0.0,1.6], [0.0, 0.6, 0.85], 6, replaceItemUniqueId=self.me_line_1_b)


        # 

        # self.me_line_2 = p.addUserDebugLine(worldLinkFramePosition, np.array(worldLinkFramePosition2) + np.array([0,0,0.1]), [0.0, 0.65, 0.55], 6)
        # replaceItemUniqueId=self.me_line_2 ???
        
        # # # print(ft_sensor_forces)
      

    def reset_internal(self):
        self._p.setGravity(0, 0, -9.81)

        self.loadEnvironment()
        self.loadRobot()
        # self.loadCameras()
        self.loadFTs()

        # TODO coordinates
        # fw = self.getFrameManager().createFrame("fw", pos=[0,0,0], orn=[0,0,0,1], ref_id=-1, scale=2.5, dirtext=[0,0.25,0.25])

        
        fc = self.getFrameManager().createFrame("fc", ref_id=self.kuka14_1, ref_link_id=14, is_body_frame=True, scale=1.4, orn=[0,1,0,0])
        fc.setVisibility(2, False)
        fc.setVisibility(4, False)

        # fee = self.getFrameManager().createFrame("fee", ref_id=self.kuka14_1, ref_link_id=9, is_body_frame=True, scale=0.5, orn=[0,0,0,1], dirtext=[-0.15,0,0.15])

        fm = self.getFrameManager().createFrame("fm", pos=[0.7,0.0,1.6], orn=[0,0.707,0,0.707], ref_id=-1, scale=0.5, dirtext=[-0.15,0,0.15])

        # fo1 = self.getFrameManager().createFrame("fo1", ref_id=self.kuka14_1, ref_link_id=-1, is_body_frame=True, scale=1.8)

        

        fs = self.getFrameManager().createFrame("fs", ref_id=self.window_id, ref_link_id=-1, is_body_frame=True, scale=0.8, orn=[0,-0.707,0,0.707])

        # fo2 = self.getFrameManager().createFrame("fo2", pos=[0,0.315,0], ref_id=fs.getFrameId(), scale=0.8, orn=[0,0.707,0,0.707])

        self.constraint_glas_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/objects/surface_glas.urdf", useFixedBase=True, globalScaling=1.0)
        p.resetBasePositionAndOrientation(self.constraint_glas_id, [0.72,0.13,1.12], [0,0,0,1])

        self.constraint_edge_glas_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/objects/edge_glas.urdf", useFixedBase=True, globalScaling=1.0)
        p.resetBasePositionAndOrientation(self.constraint_edge_glas_id, [0.72,0.13,1.12], [0,0,0,1])



        

        
        

        self.dp_frame_tx = self._p.addUserDebugParameter("tx", -0.1, 0.1, -0.08)
        self.dp_frame_ty = self._p.addUserDebugParameter("ty", -2.5, 2.5, 0)
        self.dp_frame_tz = self._p.addUserDebugParameter("tz", -2.5, 2.5, 0)
        self.dp_frame_rr = self._p.addUserDebugParameter("rr", -3.14*2, 3.14*2, 0)
        self.dp_frame_rp = self._p.addUserDebugParameter("rp", -3.14*2, 3.14*2, 3.14)
        self.dp_frame_ry = self._p.addUserDebugParameter("ry", -3.14*2, 3.14*2, 0)

        self.dp_j0 = self._p.addUserDebugParameter("j0", -6,6, 0)
        self.dp_j1 = self._p.addUserDebugParameter("j1", -6,6, 0.126)
        self.dp_j2 = self._p.addUserDebugParameter("j2", -6,6, 0)
        self.dp_j3 = self._p.addUserDebugParameter("j3", -6,6, -0.884)
        self.dp_j4 = self._p.addUserDebugParameter("j4", -6,6, 0)
        self.dp_j5 = self._p.addUserDebugParameter("j5", -6,6, 0.632)
        self.dp_j6 = self._p.addUserDebugParameter("j6", -6,6, 0)

        # Do one simulation step
        self._p.stepSimulation()

    def observation_internal(self):
        self._observation = []
        # self._observation = self.kuka14_1.getObservation()
        # gripperState = self._p.getLinkState(self.kuka14_1.kukaUid, self.kuka14_1.kukaGripperIndex)
        # gripperPos = gripperState[0]
        # gripperOrn = gripperState[1]
        # blockPos, blockOrn = self._p.getBasePositionAndOrientation(self.blockUid)

        # invGripperPos, invGripperOrn = self._p.invertTransform(gripperPos, gripperOrn)
        # gripperMat = self._p.getMatrixFromQuaternion(gripperOrn)
        # dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
        # dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
        # dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

        # gripperEul = self._p.getEulerFromQuaternion(gripperOrn)
        # #print("gripperEul")
        # #print(gripperEul)
        # blockPosInGripper, blockOrnInGripper = self._p.multiplyTransforms(invGripperPos, invGripperOrn,
        #                                                             blockPos, blockOrn)
        # projectedBlockPos2D = [blockPosInGripper[0], blockPosInGripper[1]]
        # blockEulerInGripper = self._p.getEulerFromQuaternion(blockOrnInGripper)
        # #print("projectedBlockPos2D")
        # #print(projectedBlockPos2D)
        # #print("blockEulerInGripper")
        # #print(blockEulerInGripper)

        # #we return the relative x,y position and euler angle of block in gripper space
        # blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

        # #self._p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
        # #self._p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
        # #self._p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

        # self._observation.extend(list(blockInGripperPosXYEulZ))
        return self._observation

    def render(self, mode="rgb_array", close=False):
        return np.array([]) # TODO
        # if mode != "rgb_array":
        #     return np.array([])

        # base_pos, orn = self._p.getBasePositionAndOrientation(self.kuka14_1.kukaUid)
        # view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
        #                                                         distance=self._cam_dist,
        #                                                         yaw=self._cam_yaw,
        #                                                         pitch=self._cam_pitch,
        #                                                         roll=0,
        #                                                         upAxisIndex=2)
        # proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
        #                                                 aspect=float(self.RENDER_WIDTH) / self.RENDER_HEIGHT,
        #                                                 nearVal=0.1,
        #                                                 farVal=100.0)
        # (_, _, px, _, _) = self._p.getCameraImage(width=self.RENDER_WIDTH,
        #                                         height=self.RENDER_HEIGHT,
        #                                         viewMatrix=view_matrix,
        #                                         projectionMatrix=proj_matrix,
        #                                         renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # #renderer=self._p.ER_TINY_RENDERER)

        # rgb_array = np.array(px, dtype=np.uint8)
        # rgb_array = np.reshape(rgb_array, (self.RENDER_HEIGHT, self.RENDER_WIDTH, 4))

        # rgb_array = rgb_array[:, :, :3]
        # return rgb_array

    def _termination(self):
        return False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __del__(self):
        self._p.disconnect()

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = super().env_reset
        _seed = seed
        _step = super().env_step

if __name__ == "__main__":
    tmp = ""
    if len(sys.argv) == 2:
        tmp = str(sys.argv[1])
    elif len(sys.argv) > 2:
        print("Invalid arguments!", file=sys.stderr)
        print("Usage: python3 -m gym_flexassembly.envs.flex_polishing_env [extrigger]\n")
        sys.exit(1)

    if tmp == "extrigger":
        inst = FlexPolishingEnv(stepping=False)
    else:
        inst = FlexPolishingEnv(stepping=True)
    inst.env_loop()
