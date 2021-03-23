#!/usr/bin/python3

""" MAIN: Environment for the clamp assembly scenario.
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

        # self.me_line_1 = p.addUserDebugLine([0,0,0], [0,0,0], [0.9, 0.6, 0.0], 5)
        # self.me_line_1_a = p.addUserDebugLine([0,0,0], [0,0,0], [0.9, 0.6, 0.0], 5)
        # self.me_line_1_b = p.addUserDebugLine([0,0,0], [0,0,0], [0.9, 0.6, 0.0], 5)
        # self.me_line_2 = p.addUserDebugLine([0,0,0], [0,0,0], [0.9, 0.6, 0.0], 5)

        self.seed()

        self.stage = 1

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


        self.x = -0.22
        self.y = -0.54
        self.z = 1.0
        self.qqq = [0,-1,0,0]

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

        # self.window_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/objects/window.urdf", useFixedBase=True)
        # self.window_pos = np.array([0.72,0.13,1.22])
        # p.resetBasePositionAndOrientation(self.window_id, self.window_pos, [0,0,0,1])
        # self.object_ids['window'] = self.window_id

        self.rail_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/rail.urdf", useFixedBase=True)
        self.object_ids['rail'] = self.rail_id
        p.resetBasePositionAndOrientation(self.rail_id, [-0.425,-0.7,0.72], [0,0,0,1])

        # Workpiece clamp 1
        self.clamp_1 = SpringClamp(pos=[-0.22,-0.54, 0.73], orn=[0,0,1,0])
        self.clamp_2 = SpringClamp(pos=[-0.27,-0.61, 0.73], orn=[0,0,0.0383,0.924])
        self.clamp_3 = SpringClamp(pos=[-0.34,-0.5, 0.73], orn=[0,0,0.924,0.383])
        self.clamp_4 = SpringClamp(pos=[-0.39,-0.63, 0.73], orn=[0,0,-0.5,0.866])
        

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

        self.kuka14_1 = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/epfl-iiwa14/iiwa14.urdf", useFixedBase=True)
        if self._use_real_interface:
            f = open("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/epfl-iiwa14/iiwa14.urdf","r") # TODO
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

        # self._p.resetJointState(self.kuka14_1, 1, 0.6781796865240398, 0.0)
        # self._p.resetJointState(self.kuka14_1, 2, -0.8102405282122508, 0.0)
        # self._p.resetJointState(self.kuka14_1, 3, 0.7848354497670429, 0.0)
        # self._p.resetJointState(self.kuka14_1, 4, -4.749972331779979, 0.0)
        # self._p.resetJointState(self.kuka14_1, 5, -0.6260180198993508, 0.0)
        # self._p.resetJointState(self.kuka14_1, 6, -1.0629565169913961, 0.0)
        # self._p.resetJointState(self.kuka14_1, 7, 1.6192824984624523, 0.0)

        self._p.resetJointState(self.kuka14_1, 1, 1.57086, 0.0)
        self._p.resetJointState(self.kuka14_1, 2, -0.698182, 0.0)
        self._p.resetJointState(self.kuka14_1, 3, 0.0, 0.0)
        self._p.resetJointState(self.kuka14_1, 4, 1.0769, 0.0)
        self._p.resetJointState(self.kuka14_1, 5, 0.0, 0.0)
        self._p.resetJointState(self.kuka14_1, 6, -1.32629, 0.0)
        self._p.resetJointState(self.kuka14_1, 7, 0.0, 0.0)

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

        # #lower limits for null space
        # ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # #upper limits for null space
        # ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # #joint ranges for null space
        # jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # #restposes for null space
        # rp = [1.57086,-0.698182,0.0,1.0769,0.0,-1.32629,0.0]

        # # rp = [0.6785196181667755, -0.8099947206735864, 0.7842224322887793, -4.7499723318856075, -0.6254912801606999, -1.0627425435028586, 1.618943609718446]
        # # rp = [0,0,0,0,0,0,0]
        # #joint damping coefficents
        # jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


        # # jointPoses = self._p.calculateInverseKinematics(self.kuka14_1, 9, [-0.22,-0.54,1.0], [0,-1,0,0], ll, ul, jr, rp)
        # # print(jointPoses)

        # # jointPoses = [1.57086,-0.698182,0.0,1.0769,0.0,-1.32629,0.0]
        # for i in range(7):
        #     p.resetJointState(self.kuka14_1, i+1, rp[i])


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

        # _, _, _, _, worldLinkFramePosition, worldLinkFrameOrientation = self._p.getLinkState(self.kuka14_1, 14)
        # self._p.resetBasePositionAndOrientation(self.constraint_edge_glas_id, worldLinkFramePosition, worldLinkFrameOrientation)
        # third = (np.array(worldLinkFramePosition)-np.array([0.7,0.0,1.6])) / 3.0
        # self.me_line_1 = p.addUserDebugLine(worldLinkFramePosition, np.array(worldLinkFramePosition)-third-np.array([0,0,0.05]), [0.0, 0.6, 0.85], 6, replaceItemUniqueId=self.me_line_1)
        # self.me_line_1_a = p.addUserDebugLine(np.array(worldLinkFramePosition)-third-np.array([0,0,0.05]), np.array(worldLinkFramePosition)-third-third-np.array([0,0,-0.05]), [0.0, 0.6, 0.85], 6, replaceItemUniqueId=self.me_line_1_a)
        # self.me_line_1_b = p.addUserDebugLine(np.array(worldLinkFramePosition)-third-third-np.array([0,0,-0.05]), [0.7,0.0,1.6], [0.0, 0.6, 0.85], 6, replaceItemUniqueId=self.me_line_1_b)

        #lower limits for null space
        ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        #upper limits for null space
        ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        #joint ranges for null space
        jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        rp = [1.57086,-0.698182,0.0,1.0769,0.0,-1.32629,0.0]

        # rp = [0.6785196181667755, -0.8099947206735864, 0.7842224322887793, -4.7499723318856075, -0.6254912801606999, -1.0627425435028586, 1.618943609718446]
        # rp = [0,0,0,0,0,0,0]
        #joint damping coefficents
        jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


        jointPoses = self._p.calculateInverseKinematics(self.kuka14_1, 11, [self.x,self.y,self.z], self.qqq, ll, ul, jr, rp)
        # print(jointPoses)

        if self.stage == 1:
            if self.z > 0.72:
                self.z = self.z - 0.0001
            else:
                # self.kuka7_1_egp.close_gripper(None)
                self.kuka7_1_egp.close()
                for kk in range(1000):
                    if self.kuka7_1_egp:
                        self.kuka7_1_egp.update()
                    self._p.stepSimulation()
                self.stage = 2
        elif self.stage == 2:
            if self.z < 0.8:
                self.z = self.z + 0.0001
            else:
                self.stage = 3
        elif self.stage == 3:
            if self.y > -0.685:
                self.y = self.y - 0.0001
            else:
                self.stage = 4
        elif self.stage == 4:
            if self.z > 0.75:
                self.z = self.z - 0.0001
            else:
                self.stage = 5
        elif self.stage == 5:
            # self.qqq
            # if self.z > 0.75:
            #     self.z = self.z - 0.0001
            # else:
            #     self.stage = 6
            pass
        # TODO slerp!


        p.setJointMotorControl2(self.kuka14_1,
                                1,
                                p.POSITION_CONTROL,
                                targetPosition=jointPoses[0],
                                force=1000)

        p.setJointMotorControl2(self.kuka14_1,
                                2,
                                p.POSITION_CONTROL,
                                targetPosition=jointPoses[1],
                                force=1000)

        p.setJointMotorControl2(self.kuka14_1,
                                3,
                                p.POSITION_CONTROL,
                                targetPosition=jointPoses[2],
                                force=1000)
        
        p.setJointMotorControl2(self.kuka14_1,
                                4,
                                p.POSITION_CONTROL,
                                targetPosition=jointPoses[3],
                                force=1000)

        p.setJointMotorControl2(self.kuka14_1,
                                5,
                                p.POSITION_CONTROL,
                                targetPosition=jointPoses[4],
                                force=1000)

        p.setJointMotorControl2(self.kuka14_1,
                                6,
                                p.POSITION_CONTROL,
                                targetPosition=jointPoses[5],
                                force=1000)

        p.setJointMotorControl2(self.kuka14_1,
                                7,
                                p.POSITION_CONTROL,
                                targetPosition=jointPoses[6],
                                force=1000)


        
        # self.kuka7_1_egp.close_gripper(None)

      

    def reset_internal(self):
        self._p.setGravity(0, 0, -9.81)

        self.loadEnvironment()
        self.loadRobot()
        # self.loadCameras()
        self.loadFTs()

        # TODO coordinates
        # fw = self.getFrameManager().createFrame("fw", pos=[0,0,0], orn=[0,0,0,1], ref_id=-1, scale=2.5, dirtext=[0,0.25,0.25])

        
        # fc = self.getFrameManager().createFrame("fc", ref_id=self.kuka14_1, ref_link_id=14, is_body_frame=True, scale=1.4, orn=[0,1,0,0])
        # fc.setVisibility(2, False)
        # fc.setVisibility(4, False)

        # fee = self.getFrameManager().createFrame("fee", ref_id=self.kuka14_1, ref_link_id=9, is_body_frame=True, scale=0.5, orn=[0,0,0,1], dirtext=[-0.15,0,0.15])

        # fm = self.getFrameManager().createFrame("fm", pos=[0.7,0.0,1.6], orn=[0,0.707,0,0.707], ref_id=-1, scale=0.5, dirtext=[-0.15,0,0.15])

        # fo1 = self.getFrameManager().createFrame("fo1", ref_id=self.kuka14_1, ref_link_id=-1, is_body_frame=True, scale=1.8)

        

        # fs = self.getFrameManager().createFrame("fs", ref_id=self.window_id, ref_link_id=-1, is_body_frame=True, scale=0.8, orn=[0,-0.707,0,0.707])

        # fo2 = self.getFrameManager().createFrame("fo2", pos=[0,0.315,0], ref_id=fs.getFrameId(), scale=0.8, orn=[0,0.707,0,0.707])

        # self.constraint_glas_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/objects/surface_glas.urdf", useFixedBase=True, globalScaling=1.0)
        # p.resetBasePositionAndOrientation(self.constraint_glas_id, [0.72,0.13,1.12], [0,0,0,1])

        # self.constraint_edge_glas_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/objects/edge_glas.urdf", useFixedBase=True, globalScaling=1.0)
        # p.resetBasePositionAndOrientation(self.constraint_edge_glas_id, [0.72,0.13,1.12], [0,0,0,1])


        
        

        # self.dp_frame_tx = self._p.addUserDebugParameter("tx", -0.1, 0.1, -0.08)
        # self.dp_frame_ty = self._p.addUserDebugParameter("ty", -2.5, 2.5, 0)
        # self.dp_frame_tz = self._p.addUserDebugParameter("tz", -2.5, 2.5, 0)
        # self.dp_frame_rr = self._p.addUserDebugParameter("rr", -3.14*2, 3.14*2, 0)
        # self.dp_frame_rp = self._p.addUserDebugParameter("rp", -3.14*2, 3.14*2, 3.14)
        # self.dp_frame_ry = self._p.addUserDebugParameter("ry", -3.14*2, 3.14*2, 0)

        # self.dp_j0 = self._p.addUserDebugParameter("j0", -6,6, 0)
        # self.dp_j1 = self._p.addUserDebugParameter("j1", -6,6, 0.126)
        # self.dp_j2 = self._p.addUserDebugParameter("j2", -6,6, 0)
        # self.dp_j3 = self._p.addUserDebugParameter("j3", -6,6, -0.884)
        # self.dp_j4 = self._p.addUserDebugParameter("j4", -6,6, 0)
        # self.dp_j5 = self._p.addUserDebugParameter("j5", -6,6, 0.632)
        # self.dp_j6 = self._p.addUserDebugParameter("j6", -6,6, 0)

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
