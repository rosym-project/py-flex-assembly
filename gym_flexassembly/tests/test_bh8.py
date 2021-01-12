#!/usr/bin/python

"""This is a test for spawning the IIT COMAN robot.
:Author:
  `Dennis Leroy Wigand <dwigand@cor-lab.de>`
"""

import pybullet as p
import time
import numpy as np
import math

import rospy
rospy.init_node("coman_test", anonymous=False)

# FLEX ASSEMBLY DATA IMPORTS
from gym_flexassembly import data as flexassembly_data
# print("flexassembly_data.getDataPath() = " + str(flexassembly_data.getDataPath()))

client = p.connect(p.GUI_SERVER)
p.setGravity(0, 0, -9.81, physicsClientId=client)
# p.setGravity(0, 0, 0, physicsClientId=client)

p.setAdditionalSearchPath(flexassembly_data.getDataPath())

planeId = p.loadURDF("objects/plane_solid.urdf")

bh8 = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/barrett-bh8/model.urdf", useFixedBase=True)
p.resetBasePositionAndOrientation(bh8, [0, 0, 0.6], [0,0,0,1])

_robot_map = {}
_robot_map["coman_0"] = bh8
rospy.set_param("robot_map", _robot_map)

# rail = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/rail.urdf", useFixedBase=True)

# ball = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/ball.urdf", useFixedBase=False)
# p.resetBasePositionAndOrientation(ball, [0.05, 0.005, 0.04], [0,0,0,1])

p.setTimeStep(0.001) # TODO DLW


numJoints = p.getNumJoints(bh8)
for jointIndex in range(numJoints):
  # Reset joints to the initial configuration
  p.resetJointState(bh8, jointIndex, 0)
  # p.setJointMotorControl2(bh8,
  #                         jointIndex,
  #                         p.POSITION_CONTROL,
  #                         targetPosition=cur_q[jointIndex],
  #                         force=maxForce)
  p.setJointMotorControl2(bh8,
                          jointIndex,
                          p.VELOCITY_CONTROL,
                          force=0.0)
  p.setJointMotorControl2(bh8,
                          jointIndex,
                          p.TORQUE_CONTROL,
                          force=0.0)

motorNames = []
motorIndices = []
zeroForces = []
for i in range(numJoints):
    jointInfo = p.getJointInfo(bh8, i)
    qIndex = jointInfo[3]
    if qIndex > -1:
        print("motorname " + str(jointInfo[1]) + ", index " + str(i))
        motorNames.append(str(jointInfo[1]))
        motorIndices.append(i)
        zeroForces.append(0.0)
    else:
        print("ignored joint " + str(jointInfo[1]) + ", index " + str(i))

# for i in range(len(motorIndices)):
#     p.resetJointState(bh8, motorIndices[i], cur_q[i])

while 1:
    p.stepSimulation()

    numJoints = len(motorIndices)
    # p.getNumJoints(arm.getUUid())
    jointStates = p.getJointStates(bh8, motorIndices)
    q1 = []
    qdot1 = []
    zeroAccelerations = []
    for i in range(numJoints):
      # print('i ' + str(i))
      # print('index ' + str(motorIndices[i]))
      q1.append(jointStates[i][0])
      qdot1.append(jointStates[i][1])
      zeroAccelerations.append(0)
    # q = np.array(q1)
    # qdot = np.array(qdot1)
    # # print("len qdot " + str(len(qdot)))
    # qdes = np.array([joint_pos_0,joint_pos_1,joint_pos_2,joint_pos_3,joint_pos_4,joint_pos_5,joint_pos_6])
    # # print("len qdes " + str(len(qdes)))
    # qdotdes = np.array([0,0,0,0,0,0,0])
    # # print("len qdotdes " + str(len(qdotdes)))
    # qError = qdes - q
    # # print("len qError " + str(len(qError)))
    # qdotError = qdotdes - qdot
    # # print("len qdotError " + str(len(qdotError)))
    # Kp = np.diagflat([2,2,2,2,2,2,2])
    # # print("Kp " + str(Kp))
    # Kd = np.diagflat([0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    # pp = Kp.dot(qError)
    # dd = Kd.dot(qdotError)
    # # forces = pp + dd

    # timeStep = 0.1

    M1 = p.calculateMassMatrix(bh8, q1)
    M2 = np.array(M1)
    # print(M2)
    # M = (M2 + Kd * timeStep)
    # print("POS = " + str(q1))
    # print("VEL = " + str(qdot1))
    c1 = p.calculateInverseDynamics(bh8, q1, qdot1, zeroAccelerations)
    # print("GRA = " + str(c1))
    # print(len(c1))
    c = np.array(c1)

rate = rospy.Rate(1000.0)

# while not rospy.is_shutdown():
#   # p.stepSimulation() # Only use this is we are not triggered externally...
#   # env_step()
#   # if _gui:
#   #    # time.sleep(_timeStep)
#   rate.sleep()
