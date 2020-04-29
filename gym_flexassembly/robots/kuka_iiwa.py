import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math

# import pybullet_data
import gym_flexassembly.data

class KukaIIWA:
    def __init__(self, urdfRootPath=gym_flexassembly.data.getDataPath(), timeStep=0.01, variant='7'):
        self.variant = variant
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = .35
        self.maxForce = 200.
        self.fingerAForce = 2
        self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 21
        self.useOrientation = 1
        self.kukaEndEffectorIndex = 6
        self.kukaGripperIndex = 7
        #lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        #upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        #joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        #joint damping coefficents
        self.jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001
        ]
        self.reset()

    def reset(self):
        if self.variant == '14':
            self.kukaUid = p.loadURDF(os.path.join(self.urdfRootPath, "kuka-iiwa-7/model.urdf"),flags=p.URDF_USE_INERTIA_FROM_FILE)
        else:
            self.kukaUid = p.loadURDF(os.path.join(self.urdfRootPath, "kuka-iiwa-7/model.urdf"),flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.controlMode = "JOINT_TORQUE_CONTROL"
        # self.kukaUid = objects[0]
        #for i in range (p.getNumJoints(self.kukaUid)):
        #  print(p.getJointInfo(self.kukaUid,i))
        p.resetBasePositionAndOrientation(self.kukaUid, [-0.100000, 0.000000, 0.070000],
                                      [0.000000, 0.000000, 0.000000, 1.000000])
        self.jointPositions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
            -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        ]
        self.numJoints = p.getNumJoints(self.kukaUid)
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
            p.setJointMotorControl2(self.kukaUid,
                                    jointIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=self.jointPositions[jointIndex],
                                    force=self.maxForce)

        # # self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.640000, 0.075000, -0.190000, 0.000000, 0.000000, 1.000000, 0.000000)
        # self.endEffectorPos = [0.537, 0.0, 0.5]
        # self.endEffectorAngle = 0

        self.motorNames = []
        self.motorIndices = []
        self.zeroForces = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.kukaUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                print("motorname " + str(jointInfo[1]) + ", index " + str(i))
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)
                self.zeroForces.append(0.0)

    # def getActionDimension(self):
    #     if (self.useInverseKinematics):
    #         return len(self.motorIndices)
    #     return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

    def getMotorIndices(self):
        return self.motorIndices

    def getObservationDimension(self):
        return len(self.getObservation())

    def getUUid(self):
        return self.kukaUid

    def getObservation(self):
        # observation = []
        # state = p.getLinkState(self.kukaUid, self.kukaGripperIndex)
        # p.getJointStates()
        # pos = state[0]
        # orn = state[1]
        # euler = p.getEulerFromQuaternion(orn)

        # observation.extend(list(pos))
        # observation.extend(list(euler))
        joint_states = p.getJointStates(self.kukaUid, self.motorIndices)

        # Print Debug Stuff for Joint 0
        # js_0 = joint_states[0]
        # print("len(js_0) " + str(len(js_0)))
        # q_0 = js_0[0]
        # print("q_0 " + str(q_0))
        # qd_0 = js_0[1]
        # print("qd_0 " + str(qd_0))
        # For Force Sensor
        # jointReactionForces_0 = js_0[2]
        # appliedJointMotorTorque_0 = js_0[3]
        # print("appliedJointMotorTorque_0 " + str(appliedJointMotorTorque_0))
        # q_pos[0][i] = joint_states[0][0]
        # a = joint_states[1][0]
        # print("joint_states[1][0] = " + str(joint_states[1][0]))
        # q_pos[1][i] = a

        # q_vel[0][i] = joint_states[0][1]
        # q_vel[1][i] = joint_states[1][1]

        
        return joint_states

    def getInertiaMatrix(self):
        # Controller with MassMatrix
        # https://github.com/bulletphysics/bullet3/blob/aec9968e281faca7bc56bc05ccaf0ef29d82d062/examples/pybullet/examples/pdControllerStable.py
        # https://github.com/bulletphysics/bullet3/blob/0aaae872451a69d0c93b0c8ed818667de4ad5653/examples/pybullet/gym/pybullet_utils/pd_controller_stable.py
        
        # of EEF
        dyn = p.getDynamicsInfo(self.kukaUid, -1)
        print("dyn = " + str(dyn))
        mass = dyn[0]
        friction = dyn[1]
        localInertiaDiagonal = dyn[2]
        pass

    def getGravityVector(self):
        pass

    def setControlMode(self, controlMode):
        # TODO use ENUM not String here
        if controlMode == "JOINT_IMPEDANCE_CONTROL":
            pass
        if controlMode == "JOINT_TORQUE_CONTROL":
            # # Disable the motors first
            # Disable the motors for torque control:
            p.setJointMotorControlArray(self.kukaUid,
                                        self.motorIndices,
                                        p.VELOCITY_CONTROL,
                                        forces=self.zeroForces)
            self.controlMode = controlMode
        if controlMode == "CARTESIAN_IMPEDANCE_CONTROL":
            pass

    def setCommand(self, motorCommands):
        if self.controlMode == "JOINT_TORQUE_CONTROL":
            # # Use Torque control in the loop
            # # Set the Joint Torques:
            p.setJointMotorControlArray(self.kukaUid,
                                        self.motorIndices,
                                        p.TORQUE_CONTROL,
                                        forces=motorCommands)
            # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_dynamics.py


    # def applyAction(self, motorCommands):
    #     if (self.useInverseKinematics):
    #         dx = motorCommands[0]
    #         dy = motorCommands[1]
    #         dz = motorCommands[2]
    #         da = motorCommands[3]
    #         fingerAngle = motorCommands[4]

    #         state = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
    #         actualEndEffectorPos = state[0]
    #         #print("pos[2] (getLinkState(kukaEndEffectorIndex)")
    #         #print(actualEndEffectorPos[2])

    #         self.endEffectorPos[0] = self.endEffectorPos[0] + dx
    #         if (self.endEffectorPos[0] > 0.65):
    #             self.endEffectorPos[0] = 0.65
    #         if (self.endEffectorPos[0] < 0.50):
    #             self.endEffectorPos[0] = 0.50
    #         self.endEffectorPos[1] = self.endEffectorPos[1] + dy
    #         if (self.endEffectorPos[1] < -0.17):
    #             self.endEffectorPos[1] = -0.17
    #         if (self.endEffectorPos[1] > 0.22):
    #             self.endEffectorPos[1] = 0.22
    #         #print ("self.endEffectorPos[2]")
    #         #print (self.endEffectorPos[2])
    #         #print("actualEndEffectorPos[2]")
    #         #print(actualEndEffectorPos[2])
    #         #if (dz<0 or actualEndEffectorPos[2]<0.5):
    #         self.endEffectorPos[2] = self.endEffectorPos[2] + dz
    #         self.endEffectorAngle = self.endEffectorAngle + da
    #         pos = self.endEffectorPos
    #         orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
    #         if (self.useNullSpace == 1):
    #             if (self.useOrientation == 1):
    #                 jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos,
    #                                                 orn, self.ll, self.ul, self.jr, self.rp)
    #             else:
    #                 jointPoses = p.calculateInverseKinematics(self.kukaUid,
    #                                                         self.kukaEndEffectorIndex,
    #                                                         pos,
    #                                                         lowerLimits=self.ll,
    #                                                         upperLimits=self.ul,
    #                                                         jointRanges=self.jr,
    #                                                         restPoses=self.rp)
    #         else:
    #             if (self.useOrientation == 1):
    #                 jointPoses = p.calculateInverseKinematics(self.kukaUid,
    #                                                         self.kukaEndEffectorIndex,
    #                                                         pos,
    #                                                         orn,
    #                                                         jointDamping=self.jd)
    #             else:
    #                 jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos)
    #         #print("jointPoses")
    #         #print(jointPoses)
    #         #print("self.kukaEndEffectorIndex")
    #         #print(self.kukaEndEffectorIndex)
    #         if (self.useSimulation):
    #             for i in range(self.kukaEndEffectorIndex + 1):
    #                 #print(i)
    #                 p.setJointMotorControl2(bodyUniqueId=self.kukaUid,
    #                                         jointIndex=i,
    #                                         controlMode=p.POSITION_CONTROL,
    #                                         targetPosition=jointPoses[i],
    #                                         targetVelocity=0,
    #                                         force=self.maxForce,
    #                                         maxVelocity=self.maxVelocity,
    #                                         positionGain=0.3,
    #                                         velocityGain=1)
    #         else:
    #             #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
    #             for i in range(self.numJoints):
    #                 p.resetJointState(self.kukaUid, i, jointPoses[i])
    #         #fingers
    #         p.setJointMotorControl2(self.kukaUid,
    #                                 7,
    #                                 p.POSITION_CONTROL,
    #                                 targetPosition=self.endEffectorAngle,
    #                                 force=self.maxForce)
    #         p.setJointMotorControl2(self.kukaUid,
    #                                 8,
    #                                 p.POSITION_CONTROL,
    #                                 targetPosition=-fingerAngle,
    #                                 force=self.fingerAForce)
    #         p.setJointMotorControl2(self.kukaUid,
    #                                 11,
    #                                 p.POSITION_CONTROL,
    #                                 targetPosition=fingerAngle,
    #                                 force=self.fingerBForce)

    #         p.setJointMotorControl2(self.kukaUid,
    #                                 10,
    #                                 p.POSITION_CONTROL,
    #                                 targetPosition=0,
    #                                 force=self.fingerTipForce)
    #         p.setJointMotorControl2(self.kukaUid,
    #                                 13,
    #                                 p.POSITION_CONTROL,
    #                                 targetPosition=0,
    #                                 force=self.fingerTipForce)
    #     else:
    #         for action in range(len(motorCommands)):
    #             motor = self.motorIndices[action]
    #             p.setJointMotorControl2(self.kukaUid,
    #                                     motor,
    #                                     p.POSITION_CONTROL,
    #                                     targetPosition=motorCommands[action],
    #                                     force=self.maxForce)

class KukaIIWA7(KukaIIWA):
    def __init__(self, urdfRootPath=gym_flexassembly.data.getDataPath(), timeStep=0.01):
        KukaIIWA.__init__(self, urdfRootPath=gym_flexassembly.data.getDataPath(), timeStep=0.01, variant='7')

class KukaIIWA14(KukaIIWA):
    def __init__(self, urdfRootPath=gym_flexassembly.data.getDataPath(), timeStep=0.01):
        KukaIIWA.__init__(self, urdfRootPath=gym_flexassembly.data.getDataPath(), timeStep=0.01, variant='14')

__all__ = ['KukaIIWA7', 'KukaIIWA14']