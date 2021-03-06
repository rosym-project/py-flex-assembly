"""This is the planning component.
:Author:
    `Michael Wojtynek <mwojtynek@cor-lab.de>`
    `Dennis Leroy Wigand <dwigand@cor-lab.de>`
"""

import pybullet as p
import pybullet as pi
import pybullet_data
import pybullet_planning as planning
import numpy as np
import time
import os
from pybullet_planning.utils import INF
from pybullet_planning import Pose, Point, Euler, unit_point, euler_from_quat
from pybullet_planning import BASE_LINK, RED, BLUE, GREEN
from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, has_gui, set_camera, wait_for_duration, wait_if_gui, apply_alpha
from pybullet_planning import Pose, Point, Euler
from pybullet_planning import multiply, invert, get_distance
from pybullet_planning import create_obj, create_attachment, Attachment
from pybullet_planning import link_from_name, get_link_pose, get_moving_links, get_link_name, get_disabled_collisions, \
    get_body_body_disabled_collisions, has_link, are_links_adjacent
from pybullet_planning import get_num_joints, get_joint_names, get_movable_joints, set_joint_positions, joint_from_name, \
    joints_from_names, get_sample_fn, plan_joint_motion, get_distance_fn, get_extend_fn
from pybullet_planning import dump_world, set_pose
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn, expand_links, create_box
from pybullet_planning import pairwise_collision, pairwise_collision_info, draw_collision_diagnosis, body_collision_info
from pybullet_planning.interfaces.robots import get_movable_joints, set_joint_positions, get_link_pose, get_custom_limits, get_joint_limits, body_from_name, get_body_name, get_joint_names, get_joint_positions, get_joint_q_index, get_joint
import pybullet_planning.interfaces.robots.joint as jt
from pybullet_planning.interfaces.env_manager import create_box

from pybullet_planning.interfaces.robots.body import get_bodies
# TODO clean up the imports

from pybullet_planning import inverse_kinematics, sample_tool_ik, interval_generator

import sys

# from pybullet_planning.utils import CLIENT, set_client

class FlexPlanning(object):
    def __init__(self, pybullet, robot):
        self._p = pybullet
        self._robot = robot
        ### Nullspace ###
        useNullSpace = 1 # TODO
        ikSolver = 0
        pandaEndEffectorIndex = 11 #8 TODO
        pandaNumDofs = 7 # TODO retrieve from robot

        self.ll = [-7]*pandaNumDofs
        # Upper limits for null space (todo: set them to proper range)
        self.ul = [7]*pandaNumDofs
        # Joint ranges for null space (todo: set them to proper range)
        self.jr = [7]*pandaNumDofs
        # Restposes for null space
        # self.rp = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]

        print("Planner initialized for robot with id", robot)

    def calculatePath(self, goalPosition, goalOrientation, attachments=[]):
        # Get all objects from the world and test for obstacles
        self._obstacles = []
        for body in get_bodies():
            if body != self._robot and not body in attachments:
                self._obstacles.append(body)
        p.stepSimulation()
        self.rp = get_joint_positions(self._robot, jt.get_movable_joints(self._robot)[0:7])
        print("E) Read robot with " + str(self.rp))


        # print("ALL JOINTS: " + str(jt.get_movable_joints(self._robot)))

        # Get the joint-space configuration for the desired goal position
        # goalJntPos = p.calculateInverseKinematics(self._robot, 9, goalPosition, goalOrientation, self.ll, self.ul, self.jr, self.rp, maxNumIterations=5)

        # goalJntPos = p.calculateInverseKinematics(self._robot, 12, goalPosition) # goalOrientation
        ikSolver = 0
        jd=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        jrest=[1.5,-0.1,0.0,-2.0,0.0,1.0,1.57,0.0,0.0]

        self.ll = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05]
        #upper limits for null space
        self.ul = [2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05]
        #joint ranges for null space
        self.jr = [4.8, 4.0, 4.8, 4.0, 4.8, 4.0, 6.0]

        count = 30
        goalJntPos = None
        while count > 0:
            goalJntPos = p.calculateInverseKinematics(self._robot,11,goalPosition,goalOrientation,jointDamping=jd,solver=ikSolver, maxNumIterations=10000, residualThreshold=0.0001, restPoses=jrest, lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr)
            # set_joint_positions(self._robot, jt.get_movable_joints(self._robot), goalJntPos)
            for jj in range(len(jt.get_movable_joints(self._robot))):
                p.resetJointState(self._robot, jt.get_movable_joints(self._robot)[jj], goalJntPos[jj])
            p.stepSimulation()
            count = count - 1

        print("GOAL POS: " + str(goalJntPos))
 
        # Plan the a coolision-free path
        # path = planning.plan_joint_motion(self._robot, jt.get_movable_joints(self._robot)[0:7], goalJntPos[0:7], obstacles=self._obstacles, attachments=attachments)
        path = None
        path = planning.plan_joint_motion(self._robot, jt.get_movable_joints(self._robot)[0:7], goalJntPos[0:7])

        if path is None:
            print("\nNo plan found!\n", file=sys.stderr)
        else:
            print("\nA motion plan is found!\n")

        return path, goalJntPos[0:7]

    def getInvolvedRobotJoints(self):
        return jt.get_movable_joints(self._robot)[0:7]

    def getInvolvedRobotJointNames(self):
        return get_joint_names(self._robot, self.getInvolvedRobotJoints())

    def updateRobotConfiguration(self, robot_id, config):
        set_joint_positions(robot_id, jt.get_movable_joints(robot_id), config)
        # print("S) Update robot " + str(robot_id) + " with " + str(config) + " : TO = " + str(get_joint_positions(robot_id, jt.get_movable_joints(self._robot))[0:7]))
        p.stepSimulation()
        print("S) Read robot with " + str(get_joint_positions(robot_id, jt.get_movable_joints(robot_id)[0:7])))

    def updateObjectPoses(self, object_id, pos, orn):
        for body in get_bodies():
            if body == object_id:
                p.resetBasePositionAndOrientation(body, pos, orn)
                return
        print("Warning, object not found: " + str(object_id) + "!")

