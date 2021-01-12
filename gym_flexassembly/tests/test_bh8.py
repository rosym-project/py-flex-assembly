#!/usr/bin/python

"""This is a test for spawning the IIT COMAN robot.
:Author:
  `Dennis Leroy Wigand <dwigand@cor-lab.de>`
"""

import pybullet as p
import time
import numpy as np
import math

# import rospy
# rospy.init_node("coman_test", anonymous=False)

# FLEX ASSEMBLY DATA IMPORTS
from gym_flexassembly import data as flexassembly_data
# print("flexassembly_data.getDataPath() = " + str(flexassembly_data.getDataPath()))

client = p.connect(p.GUI_SERVER)
p.setGravity(0, 0, -9.81, physicsClientId=client)
# p.setGravity(0, 0, 0, physicsClientId=client)

p.setAdditionalSearchPath(flexassembly_data.getDataPath())

planeId = p.loadURDF("objects/plane_solid.urdf")

table = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/objects/table_2/table_2.urdf", useFixedBase=True)
# p.resetBasePositionAndOrientation(table, [-0.69, -0.5, 0], [0,0,1,0])
p.resetBasePositionAndOrientation(table, [0.83, 0.5, 0], [0,0,0,1])

left_arm = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/epfl-iiwa14-bh8/model.urdf", useFixedBase=True)
p.resetBasePositionAndOrientation(left_arm, [0, -0.7, 0.4], [0,0,0,1])

right_arm = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/epfl-iiwa14-bh8/model.urdf", useFixedBase=True)
p.resetBasePositionAndOrientation(right_arm, [0, 0.7, 0.4], [0,0,0,1])

box = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/cube.urdf", useFixedBase=False)
p.resetBasePositionAndOrientation(box, [0.15, 0, 0.803], [0,0,0,1])



# _robot_map = {}
# _robot_map["coman_0"] = left_arm
# rospy.set_param("robot_map", _robot_map)

# rail = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/rail.urdf", useFixedBase=True)

# ball = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/ball.urdf", useFixedBase=False)
# p.resetBasePositionAndOrientation(ball, [0.05, 0.005, 0.04], [0,0,0,1])

p.setTimeStep(0.001) # TODO DLW

maxForce = 1000

numJoints = p.getNumJoints(left_arm)
for jointIndex in range(numJoints):
  # Reset joints to the initial configuration
  # p.resetJointState(left_arm, jointIndex, 0)
  # p.setJointMotorControl2(left_arm,
  #                         jointIndex,
  #                         p.POSITION_CONTROL,
  #                         targetPosition=0,
  #                         force=maxForce)
  # p.setJointMotorControl2(left_arm,
  #                         jointIndex,
  #                         p.VELOCITY_CONTROL,
  #                         force=0.0)
  # p.setJointMotorControl2(left_arm,
  #                         jointIndex,
  #                         p.TORQUE_CONTROL,
  #                         force=0.0)
  pass

l0 = p.addUserDebugParameter("l0", -2.7, 2.7, 1.251)
l1 = p.addUserDebugParameter("l1", -2.7, 2.7, 0.540)
l2 = p.addUserDebugParameter("l2", -2.7, 2.7, 0.114)
l3 = p.addUserDebugParameter("l3", -2.7, 2.7, -0.966)
l4 = p.addUserDebugParameter("l4", -2.7, 2.7, -0.028)
l5 = p.addUserDebugParameter("l5", -2.7, 2.7, 1.620)
l6 = p.addUserDebugParameter("l6", -2.7, 2.7, -0.171)
l7 = p.addUserDebugParameter("lS", 0, 3.14, 0)
l8 = p.addUserDebugParameter("lG", 0, 2.7, 0)

r0 = p.addUserDebugParameter("r0", -2.7, 2.7, -1.279)
r1 = p.addUserDebugParameter("r1", -2.7, 2.7, 0.739)
r2 = p.addUserDebugParameter("r2", -2.7, 2.7, -0.142)
r3 = p.addUserDebugParameter("r3", -2.7, 2.7, -0.625)
r4 = p.addUserDebugParameter("r4", -2.7, 2.7, -0.085)
r5 = p.addUserDebugParameter("r5", -2.7, 2.7, 1.819)
r6 = p.addUserDebugParameter("r6", -2.7, 2.7, 0.341)
r7 = p.addUserDebugParameter("rS", 0, 3.14, 0)
r8 = p.addUserDebugParameter("rG", 0, 2.7, 0)

x = p.addUserDebugParameter("X", -0.3, 0.3, 0)
z = p.addUserDebugParameter("Z", 0.8, 1.1, 1.1)
d = p.addUserDebugParameter("D", 0, 0.5, 0.2)

motorNames = []
motorIndices = []
zeroForces = []
for i in range(numJoints):
    jointInfo = p.getJointInfo(left_arm, i)
    qIndex = jointInfo[3]
    if qIndex > -1:
        print("motorname " + str(jointInfo[1]) + ", index " + str(i))
        motorNames.append(str(jointInfo[1]))
        motorIndices.append(i)
        zeroForces.append(0.0)
    else:
        print("ignored joint " + str(jointInfo[1]) + ", index " + str(i))

# for i in range(len(motorIndices)):
#     p.resetJointState(left_arm, motorIndices[i], cur_q[i])

count = 0

while 1:
    p.stepSimulation()

    if count < 5000:
      vX = p.readUserDebugParameter(x)
      vZ = p.readUserDebugParameter(z)
      vD = p.readUserDebugParameter(d)
      vL8 = p.readUserDebugParameter(l8)
      count = count + 1
    elif count < 10000:
      vX = 0.148
      vZ = 0.893
      vD = 0.108
      vL8 = p.readUserDebugParameter(l8)
      count = count + 1
    elif count < 15000:
      vL8 = 1.123
      count = count + 1
    elif count == 15000:
      if vX > -0.1:
        vX = vX - 0.0001
      else:
        count = 20000
    elif count == 20000:
      if vX < 0.3:
        vX = vX + 0.0001
      else:
        count = 15000

    
    # vL0 = p.readUserDebugParameter(l0)
    # p.setJointMotorControl2(left_arm,
    #                       1,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL0,
    #                       force=maxForce)
    # vL1 = p.readUserDebugParameter(l1)
    # p.setJointMotorControl2(left_arm,
    #                       2,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL1,
    #                       force=maxForce)
    # vL2 = p.readUserDebugParameter(l2)
    # p.setJointMotorControl2(left_arm,
    #                       3,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL2,
    #                       force=maxForce)
    # vL3 = p.readUserDebugParameter(l3)
    # p.setJointMotorControl2(left_arm,
    #                       4,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL3,
    #                       force=maxForce)
    # vL4 = p.readUserDebugParameter(l4)
    # p.setJointMotorControl2(left_arm,
    #                       5,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL4,
    #                       force=maxForce)
    # vL5 = p.readUserDebugParameter(l5)
    # p.setJointMotorControl2(left_arm,
    #                       6,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL5,
    #                       force=maxForce)
    # vL6 = p.readUserDebugParameter(l6)
    # p.setJointMotorControl2(left_arm,
    #                       7,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL6,
    #                       force=maxForce)
    vL7 = p.readUserDebugParameter(l7)
    p.setJointMotorControl2(left_arm,
                          12,
                          p.POSITION_CONTROL,
                          targetPosition=vL7,
                          force=maxForce)
    p.setJointMotorControl2(left_arm,
                          16,
                          p.POSITION_CONTROL,
                          targetPosition=vL7,
                          force=maxForce)

    
    # # if count >= 1000:
    # #   vL8 = 0.9
    # # else:
    # #   p.resetBasePositionAndOrientation(box, [0.15, 0, 0.79], [0,0,0,1])
    # #   vL8 = 0.0
    p.setJointMotorControl2(left_arm,
                          13,
                          p.POSITION_CONTROL,
                          targetPosition=vL8,
                          force=maxForce)
    p.setJointMotorControl2(left_arm,
                          14,
                          p.POSITION_CONTROL,
                          targetPosition=vL8*0.33,
                          force=maxForce)

    p.setJointMotorControl2(left_arm,
                          17,
                          p.POSITION_CONTROL,
                          targetPosition=vL8,
                          force=maxForce)
    p.setJointMotorControl2(left_arm,
                          18,
                          p.POSITION_CONTROL,
                          targetPosition=vL8*0.33,
                          force=maxForce)

    p.setJointMotorControl2(left_arm,
                          20,
                          p.POSITION_CONTROL,
                          targetPosition=vL8,
                          force=maxForce)
    p.setJointMotorControl2(left_arm,
                          21,
                          p.POSITION_CONTROL,
                          targetPosition=vL8*0.33,
                          force=maxForce)

    # # RIGHT
    # vR0 = p.readUserDebugParameter(r0)
    # p.setJointMotorControl2(right_arm,
    #                       1,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vR0,
    #                       force=maxForce)
    # vR1 = p.readUserDebugParameter(r1)
    # p.setJointMotorControl2(right_arm,
    #                       2,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vR1,
    #                       force=maxForce)
    # vR2 = p.readUserDebugParameter(r2)
    # p.setJointMotorControl2(right_arm,
    #                       3,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vR2,
    #                       force=maxForce)
    # vR3 = p.readUserDebugParameter(r3)
    # p.setJointMotorControl2(right_arm,
    #                       4,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vR3,
    #                       force=maxForce)
    # vR4 = p.readUserDebugParameter(r4)
    # p.setJointMotorControl2(right_arm,
    #                       5,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vR4,
    #                       force=maxForce)
    # vR5 = p.readUserDebugParameter(r5)
    # p.setJointMotorControl2(right_arm,
    #                       6,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vR5,
    #                       force=maxForce)
    # vR6 = p.readUserDebugParameter(r6)
    # p.setJointMotorControl2(right_arm,
    #                       7,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vR6,
    #                       force=maxForce)
    vR7 = p.readUserDebugParameter(r7)
    p.setJointMotorControl2(right_arm,
                          12,
                          p.POSITION_CONTROL,
                          targetPosition=vR7,
                          force=maxForce)
    p.setJointMotorControl2(right_arm,
                          16,
                          p.POSITION_CONTROL,
                          targetPosition=vR7,
                          force=maxForce)
    
    # vR8 = p.readUserDebugParameter(r8)
    # # if count >= 1000:
    # #   vR8 = 0.9
    # # else:
    # #   vR8 = 0.0
    vR8 = vL8
    p.setJointMotorControl2(right_arm,
                          13,
                          p.POSITION_CONTROL,
                          targetPosition=vR8,
                          force=maxForce)
    p.setJointMotorControl2(right_arm,
                          14,
                          p.POSITION_CONTROL,
                          targetPosition=vR8*0.33,
                          force=maxForce)

    p.setJointMotorControl2(right_arm,
                          17,
                          p.POSITION_CONTROL,
                          targetPosition=vR8,
                          force=maxForce)
    p.setJointMotorControl2(right_arm,
                          18,
                          p.POSITION_CONTROL,
                          targetPosition=vR8*0.33,
                          force=maxForce)

    p.setJointMotorControl2(right_arm,
                          20,
                          p.POSITION_CONTROL,
                          targetPosition=vR8,
                          force=maxForce)
    p.setJointMotorControl2(right_arm,
                          21,
                          p.POSITION_CONTROL,
                          targetPosition=vR8*0.33,
                          force=maxForce)

    numJoints = len(motorIndices)
    # p.getNumJoints(arm.getUUid())
    jointStates = p.getJointStates(left_arm, motorIndices)
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

    M1 = p.calculateMassMatrix(left_arm, q1)
    M2 = np.array(M1)
    # print(M2)
    # M = (M2 + Kd * timeStep)
    # print("POS = " + str(q1))
    # print("VEL = " + str(qdot1))
    c1 = p.calculateInverseDynamics(left_arm, q1, qdot1, zeroAccelerations)
    # print("GRA = " + str(c1))
    # print(len(c1))
    c = np.array(c1)

    #lower limits for null space
    ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    #upper limits for null space
    ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    #joint ranges for null space
    jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    #restposes for null space
    rp = [1.251,0.540,0.114,-0.966,-0.028,1.620,-0.171]
    #joint damping coefficents
    jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # vX = p.readUserDebugParameter(x)
    # vZ = p.readUserDebugParameter(z)
    # vD = p.readUserDebugParameter(d)

 

    jointPoses = p.calculateInverseKinematics(left_arm, 11, [vX, -vD, vZ], [-1,1,0,0], ll, ul, jr, rp)

    rpR = [-1.279,0.739,-0.142,-0.625,-0.085,1.819,0.341]
    jointPosesR = p.calculateInverseKinematics(right_arm, 11, [vX, vD, vZ], [1,-1,0,0], ll, ul, jr, rpR)

    p.setJointMotorControl2(left_arm,
                          1,
                          p.POSITION_CONTROL,
                          targetPosition=jointPoses[0],
                          force=maxForce)
    p.setJointMotorControl2(left_arm,
                          2,
                          p.POSITION_CONTROL,
                          targetPosition=jointPoses[1],
                          force=maxForce)
    p.setJointMotorControl2(left_arm,
                          3,
                          p.POSITION_CONTROL,
                          targetPosition=jointPoses[2],
                          force=maxForce)
    p.setJointMotorControl2(left_arm,
                          4,
                          p.POSITION_CONTROL,
                          targetPosition=jointPoses[3],
                          force=maxForce)
    p.setJointMotorControl2(left_arm,
                          5,
                          p.POSITION_CONTROL,
                          targetPosition=jointPoses[4],
                          force=maxForce)
    p.setJointMotorControl2(left_arm,
                          6,
                          p.POSITION_CONTROL,
                          targetPosition=jointPoses[5],
                          force=maxForce)
    p.setJointMotorControl2(left_arm,
                          7,
                          p.POSITION_CONTROL,
                          targetPosition=jointPoses[6],
                          force=maxForce)


    p.setJointMotorControl2(right_arm,
                          1,
                          p.POSITION_CONTROL,
                          targetPosition=jointPosesR[0],
                          force=maxForce)
    p.setJointMotorControl2(right_arm,
                          2,
                          p.POSITION_CONTROL,
                          targetPosition=jointPosesR[1],
                          force=maxForce)
    p.setJointMotorControl2(right_arm,
                          3,
                          p.POSITION_CONTROL,
                          targetPosition=jointPosesR[2],
                          force=maxForce)
    p.setJointMotorControl2(right_arm,
                          4,
                          p.POSITION_CONTROL,
                          targetPosition=jointPosesR[3],
                          force=maxForce)
    p.setJointMotorControl2(right_arm,
                          5,
                          p.POSITION_CONTROL,
                          targetPosition=jointPosesR[4],
                          force=maxForce)
    p.setJointMotorControl2(right_arm,
                          6,
                          p.POSITION_CONTROL,
                          targetPosition=jointPosesR[5],
                          force=maxForce)
    p.setJointMotorControl2(right_arm,
                          7,
                          p.POSITION_CONTROL,
                          targetPosition=jointPosesR[6],
                          force=maxForce)

    # p.setJointMotorControl2(left_arm,
    #                       12,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL7,
    #                       force=maxForce)
    # p.setJointMotorControl2(left_arm,
    #                       16,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL7,
    #                       force=maxForce)

    # p.setJointMotorControl2(left_arm,
    #                       13,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL8,
    #                       force=maxForce)
    # p.setJointMotorControl2(left_arm,
    #                       14,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL8*0.33,
    #                       force=maxForce)

    # p.setJointMotorControl2(left_arm,
    #                       17,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL8,
    #                       force=maxForce)
    # p.setJointMotorControl2(left_arm,
    #                       18,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL8*0.33,
    #                       force=maxForce)

    # p.setJointMotorControl2(left_arm,
    #                       20,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL8,
    #                       force=maxForce)
    # p.setJointMotorControl2(left_arm,
    #                       21,
    #                       p.POSITION_CONTROL,
    #                       targetPosition=vL8*0.33,
    #                       force=maxForce)

    

# rate = rospy.Rate(1000.0)

# while not rospy.is_shutdown():
#   # p.stepSimulation() # Only use this is we are not triggered externally...
#   # env_step()
#   # if _gui:
#   #    # time.sleep(_timeStep)
#   rate.sleep()
