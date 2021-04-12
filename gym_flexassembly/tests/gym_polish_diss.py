import os, inspect

import pybullet as p
import time
import math
import numpy as np
# from pyquaternion import Quaternion

# import data

# from constraints import frame
# from constraints import constraint_manager

# import trimesh

import signal

import sys
print(sys.path)

# try:
#     from .robots import KukaIIWA7
# except (ImportError, SystemError):
#     from robots import KukaIIWA7


def getRayFromTo(mouseX, mouseY):
    width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera(
    )
    camPos = [
        camTarget[0] - dist * camForward[0], camTarget[1] - dist * camForward[1],
        camTarget[2] - dist * camForward[2]
    ]
    farPlane = 10000
    rayForward = [(camTarget[0] - camPos[0]), (camTarget[1] - camPos[1]), (camTarget[2] - camPos[2])]
    invLen = farPlane * 1. / (math.sqrt(rayForward[0] * rayForward[0] + rayForward[1] *
                                        rayForward[1] + rayForward[2] * rayForward[2]))
    rayForward = [invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]]
    rayFrom = camPos
    oneOverWidth = float(1) / float(width)
    oneOverHeight = float(1) / float(height)
    dHor = [horizon[0] * oneOverWidth, horizon[1] * oneOverWidth, horizon[2] * oneOverWidth]
    dVer = [vertical[0] * oneOverHeight, vertical[1] * oneOverHeight, vertical[2] * oneOverHeight]
    rayToCenter = [
        rayFrom[0] + rayForward[0], rayFrom[1] + rayForward[1], rayFrom[2] + rayForward[2]
    ]
    rayTo = [
        rayToCenter[0] - 0.5 * horizon[0] + 0.5 * vertical[0] + float(mouseX) * dHor[0] -
        float(mouseY) * dVer[0], rayToCenter[1] - 0.5 * horizon[1] + 0.5 * vertical[1] +
        float(mouseX) * dHor[1] - float(mouseY) * dVer[1], rayToCenter[2] - 0.5 * horizon[2] +
        0.5 * vertical[2] + float(mouseX) * dHor[2] - float(mouseY) * dVer[2]
    ]
    return rayFrom, rayTo

# Open server and changing background color in visualizer
p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')

################################
### Simulation configuration ###
################################

# Setting gravity
p.setGravity(0,0,-9.81)

p.setTimeStep(0.001)
# p.setRealTimeSimulation(1)

#########################################################
### Setting up the environment for the polishing task ###
#########################################################

# Getting pybullet assets
import pybullet_data
print("pybullet_data.getDataPath() = " + str(pybullet_data.getDataPath()))
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Disable rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# Floor SHOULD BE ALWAYS ID 0
p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/objects/plane_solid.urdf", useMaximalCoordinates=True) # Brauche ich fuer die hit rays

window_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/objects/window.urdf", useFixedBase=True)
window_pos = np.array([0.75,0,1.1])
p.resetBasePositionAndOrientation(window_id, window_pos, [0,0,0,1])

##########################
###   Draw Traj Path   ###
##########################

start = window_pos + np.array([-0.01,0.3,0.52])
robot_start = start
end = start + np.array([0,-0.6,0])
p.addUserDebugLine(start, end, [0.3, 0.3, 0.3], 2)
# dir
p.addUserDebugLine(start+0.5*(end-start), start+0.5*(end-start)+[0,0.03,0.03], [0.3, 0.3, 0.3], 2)
p.addUserDebugLine(start+0.5*(end-start), start+0.5*(end-start)+[0,0.03,-0.03], [0.3, 0.3, 0.3], 2)

for i in range(0,2):
    start = end
    end = end + np.array([0,0,-0.18])
    p.addUserDebugLine(start, end, [0.3, 0.3, 0.3], 2)

    start = end
    if i % 2:
        end = end + np.array([0,-0.6,0])
        dir1 = [0,0.03,0.03]
        dir2 = [0,0.03,-0.03]
    else:
        end = end + np.array([0,0.6,0])
        dir1 = [0,-0.03,0.03]
        dir2 = [0,-0.03,-0.03]
    p.addUserDebugLine(start, end, [0.3, 0.3, 0.3], 2)
    # dir
    p.addUserDebugLine(start+0.5*(end-start), start+0.5*(end-start)+dir1, [0.3, 0.3, 0.3], 2)
    p.addUserDebugLine(start+0.5*(end-start), start+0.5*(end-start)+dir2, [0.3, 0.3, 0.3], 2)

# ux = p.addUserDebugLine([0, 0, 0], [0,0,0], [0.3, 0.3, 0.3], 2)
# uy = p.addUserDebugLine([0, 0, 0], [0,0,0], [0.3, 0.3, 0.3], 2)
# uz = p.addUserDebugLine([0, 0, 0], [0,0,0], [0.3, 0.3, 0.3], 2)

# # Workpiece to be polished
# workpiece_id = p.loadURDF("/home/dwigand/code/cogimon/CoSimA/pyBullet/pyCompliantInteractionPlanning/gym_flexassembly/data/3d/bend_wood.urdf", useFixedBase=True, flags = p.URDF_USE_INERTIA_FROM_FILE)
# wood_offset_table_x = -0.88
# wood_offset_table_y = -0.22
# wood_offset_table_z = -0.73
# wood_offset_world = [wood_offset_table_x, wood_offset_table_y, wood_offset_table_z]
# p.resetBasePositionAndOrientation(workpiece_id, wood_offset_world, [0,0,0,1])

# Getting my data assets
# urdfRootPath = data.getDataPath()
# frame_ghost_id = p.loadURDF(os.path.join(urdfRootPath, "frame.urdf"), useFixedBase=True)

# Enable rendering again
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

################################
### KUKA MOCK ###
################################
kmr_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/kuka-kmr/model.urdf", useFixedBase=True)
p.resetBasePositionAndOrientation(kmr_id, [-0.17,-0.36,0], [0,0,0.757,0.757])

# arm = KukaIIWA7()
arm_id = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/epfl-iiwa14/iiwa14.urdf", useFixedBase=True)
p.resetBasePositionAndOrientation(arm_id, [0,0,0.7], [0,0,0,1])
# arm.setControlMode("JOINT_TORQUE_CONTROL")
#                           0x010
# collisionFilterGroup_kuka = 0x10
# #                           0x001
# collisionFilterMask_kuka = 0x1
# for i in range(p.getNumJoints(arm_id)):
#     p.setCollisionFilterGroupMask(arm_id, i-1, collisionFilterGroup_kuka, collisionFilterMask_kuka)

# p.enableJointForceTorqueSensor(arm_id, 7)
p.enableJointForceTorqueSensor(arm_id, 8) # Why 8?

arm_ft_7 = p.addUserDebugLine([0, 0, 0], [0, 0, 0], [0.6, 0.3, 0.1], parentObjectUniqueId=arm_id, parentLinkIndex=7)

##########################
### Debug GUI elements ###
##########################

# # Test Frames!
# first_frame1 = frame.Frame(p, "first_frame1")
# first_frame1.resetPositionAndOrientation([0, 0.1, 1], [0,1,0,0])

# first_frame2 = frame.Frame(p, "first_frame2")
# first_frame2.resetPositionAndOrientation([1, 1, 3], [0,0,0,1])

# ux = p.addUserDebugLine([0, 0, 0], [0,0,0], [0.3, 0.3, 0.3], 2)
# uy = p.addUserDebugLine([0, 0, 0], [0,0,0], [0.3, 0.3, 0.3], 2)
# uz = p.addUserDebugLine([0, 0, 0], [0,0,0], [0.3, 0.3, 0.3], 2)

dp_frame_tx = p.addUserDebugParameter("tx", -0.1, 0.1, -0.08)
dp_frame_ty = p.addUserDebugParameter("ty", -2.5, 2.5, 0)
dp_frame_tz = p.addUserDebugParameter("tz", -2.5, 2.5, 0)
dp_frame_rr = p.addUserDebugParameter("rr", -3.14*2, 3.14*2, 0)
dp_frame_rp = p.addUserDebugParameter("rp", -3.14*2, 3.14*2, 3.14)
dp_frame_ry = p.addUserDebugParameter("ry", -3.14*2, 3.14*2, 0)


#################
### Main loop ###
#################

# objtrimesh = trimesh.load_mesh("/home/dwigand/code/cogimon/CoSimA/pyBullet/pyCompliantInteractionPlanning/gym_flexassembly/data/3d/bend_wood_y_z.obj")
# objtrimesh.apply_scale(0.1)
# Need to be done to be conform with bullet loading the model
# objtrimesh.apply_transform(trimesh.transformations.euler_matrix(1.57, 0, 0, 'rxyz'))
# 
# for face_index in range(len(objtrimesh.face_normals)):
#     # print(objtrimesh.vertices[point_index])
#     p.addUserDebugLine(objtrimesh.vertices[objtrimesh.faces[face_index][0]]+wood_offset_world, objtrimesh.vertices[objtrimesh.faces[face_index][1]]+wood_offset_world, [0.3, 0.3, 0.3], 3)
#     p.addUserDebugLine(objtrimesh.vertices[objtrimesh.faces[face_index][1]]+wood_offset_world, objtrimesh.vertices[objtrimesh.faces[face_index][2]]+wood_offset_world, [0.3, 0.3, 0.3], 3)
#     p.addUserDebugLine(objtrimesh.vertices[objtrimesh.faces[face_index][2]]+wood_offset_world, objtrimesh.vertices[objtrimesh.faces[face_index][0]]+wood_offset_world, [0.3, 0.3, 0.3], 3)

#     cgx = (objtrimesh.vertices[objtrimesh.faces[face_index][0]][0] + objtrimesh.vertices[objtrimesh.faces[face_index][1]][0] + objtrimesh.vertices[objtrimesh.faces[face_index][2]][0])/3
#     cgy = (objtrimesh.vertices[objtrimesh.faces[face_index][0]][1] + objtrimesh.vertices[objtrimesh.faces[face_index][1]][1] + objtrimesh.vertices[objtrimesh.faces[face_index][2]][1])/3
#     cgz = (objtrimesh.vertices[objtrimesh.faces[face_index][0]][2] + objtrimesh.vertices[objtrimesh.faces[face_index][1]][2] + objtrimesh.vertices[objtrimesh.faces[face_index][2]][2])/3

#     p.addUserDebugLine(np.array([cgx, cgy, cgz])+wood_offset_world, objtrimesh.face_normals[face_index]*0.1+np.array([cgx, cgy, cgz])+wood_offset_world, [0.3, 0.3, 1], 3)


# MyKey1 = p.addUserData(first_frame2.getFrameId(), "MyKey1", "MyValue1")

# cm = constraint_manager.ConstraintManager(p)
# cm.addFrame(first_frame1)
# cm.addFrame(first_frame2)

# numJoints = p.getNumJoints(arm_id)

# p.setRealTimeSimulation(1)

count_draw = 0

p.stepSimulation()

once = False

segment = 1
while (1):
    pose_frame_tx = p.readUserDebugParameter(dp_frame_tx)
    pose_frame_ty = p.readUserDebugParameter(dp_frame_ty)
    pose_frame_tz = p.readUserDebugParameter(dp_frame_tz)
    pose_frame_rr = p.readUserDebugParameter(dp_frame_rr)
    pose_frame_rp = p.readUserDebugParameter(dp_frame_rp)
    pose_frame_ry = p.readUserDebugParameter(dp_frame_ry)

    # camData = p.getDebugVisualizerCamera()
    # viewMat = camData[2]
    # projMat = camData[3]
    # p.getCameraImage(256,
    #                 256,
    #                 viewMatrix=viewMat,
    #                 projectionMatrix=projMat,
    #                 renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # p.resetBasePositionAndOrientation(first_frame1.getFrameId(), [pose_frame_tx, pose_frame_ty, pose_frame_tz], p.getQuaternionFromEuler([pose_frame_rr, pose_frame_rp, pose_frame_ry]))

    # f1_pos, f1_orn = p.getBasePositionAndOrientation(first_frame1.getFrameId())


    # xxx = xxx + (target-xxx) * 0.001
    # if segment > 0:
    #     # hinweg
    #     if segment == 1:
    #         if xxx 


    # xxx = robot_start
    # end = robot_xxx + np.array([0,-0.6,0])
    # p.addUserDebugLine(robot_xxx, end, [0.3, 0.3, 0.3], 2)

    # for i in range(0,4):
    #     robot_xxx = end
    #     end = end + np.array([0,0,-0.15])
    #     p.addUserDebugLine(robot_xxx, end, [0.3, 0.3, 0.3], 2)

    #     robot_xxx = end
    #     if i % 2:
    #         end = end + np.array([0,-0.6,0])
    #     else:
    #         end = end + np.array([0,0.6,0])
    #     p.addUserDebugLine(robot_xxx, end, [0.3, 0.3, 0.3], 2)

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

  


    jointPoses = p.calculateInverseKinematics(arm_id, 11, robot_start + np.array([pose_frame_tx,pose_frame_ty,pose_frame_tz]), [0.757,0,0.757,0],ll, ul, jr, rp) # muss 11 sein
    maxForce = 1000.0
    if not once:
        # once = True
        for jointIndex in range(1,7):
            # p.resetJointState(arm_id, jointIndex, jointPoses[jointIndex-1])
            p.setJointMotorControl2(arm_id,
                                    jointIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=jointPoses[jointIndex-1],
                                    force=maxForce)
        p.setJointMotorControl2(arm_id,
                                    12,
                                    p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=90000.0)
        p.setJointMotorControl2(arm_id,
                                    13,
                                    p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=90000.0)
            


    # Get force
    # print("ft7 = " + str(p.getJointState(arm_id, 7)[2][0:3]))
    # print("ft8 = " + str(p.getJointState(arm_id, 8)[2][0:3]))
    # force = np.array(p.getJointState(arm_id, 8)[2][0:3])
    # force = force/np.linalg.norm(force)
    
    # arm_ft_7 = p.addUserDebugLine([0, 0, 0], force, [0.6, 0.3, 0.1], parentObjectUniqueId=arm_id, parentLinkIndex=7, replaceItemUniqueId = arm_ft_7)

    # p.addUserDebugLine([0, 0, 0], [0,0,0.1], [0.6, 0.3, 0.1], parentObjectUniqueId=arm_id, parentLinkIndex=11, lifeTime=3)

    # _,_,_,_,worldLinkFramePosition,_ = p.getLinkState(arm_id,11)
    # p.addUserDebugLine(worldLinkFramePosition, worldLinkFramePosition+np.array([0,0,0.001]), [0.6, 0.3, 0.1], 4, lifeTime=0.4)


    contacts = p.getContactPoints(bodyA=arm_id, bodyB=window_id, linkIndexB=0)
    # contactFlag, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, positionOnA, positionOnB, contactNormalOnB, contactDistance, normalForce, lateralFriction1, lateralFrictionDir1, lateralFriction2, lateralFrictionDir2
    for contact in contacts:
        if contact[3] == 12:
            # link_index = contact[3]
            # print(link_index)
            ddd = 40000.0
            p.addUserDebugLine(contact[6], np.array(contact[6]) + np.array(contact[7])*0.001*contact[9]*0.005, [contact[9]/ddd, 0, (ddd-contact[9])/ddd], 4)
            # print((ddd-contact[9])/ddd)
        elif contact[3] == 13:
            # link_index = contact[3]
            # print(link_index)
            ddd = 40000.0
            p.addUserDebugLine(contact[6], np.array(contact[6]) + np.array(contact[7])*0.001*contact[9]*0.005, [contact[9]/ddd, 0, (ddd-contact[9])/ddd], 4)
            # print((ddd-contact[9])/ddd)

    # keys = p.getKeyboardEvents()
    # mouseEvents = p.getMouseEvents()
    
    # for e in mouseEvents:
    #     # Selection
    #     if ((e[0] == 2) and (e[4] & p.KEY_WAS_TRIGGERED)):
    #         if e[3] == 0: # Left click
    #             ctrl_pressed = False
    #             for k, v in keys.items():
    #                 if not (k == p.B3G_CONTROL and (v & p.KEY_WAS_TRIGGERED)):
    #                     ctrl_pressed = True
    #             if not ctrl_pressed:
    #                 mouseX = e[1]
    #                 mouseY = e[2]
    #                 rayFrom, rayTo = getRayFromTo(mouseX, mouseY)
    #                 rayInfo = p.rayTest(rayFrom, rayTo)
    #                 # print("rayyyyyy " + str(rayInfo))
    #                 cm.handlePick(rayInfo)
    #         elif e[3] == 2: # Right click
    #             selectedFrame = cm.getSelectedFrame()
    #             if selectedFrame:
    #                 mouseX = e[1]
    #                 mouseY = e[2]
    #                 rayFrom, rayTo = getRayFromTo(mouseX, mouseY)
    #                 rayInfo = p.rayTest(rayFrom, rayTo)
    #                 if rayInfo and len(rayInfo) > 0 and rayInfo[0][0] > 0 and selectedFrame.getFrameId() != rayInfo[0][0]:
    #                     v_source = np.array([0,0,1])
    #                     v_normal = np.array(rayInfo[0][4])

    #                     v_source_norm = v_source/np.linalg.norm(v_source)
    #                     v_normal_norm = v_normal/np.linalg.norm(v_normal)

    #                     cos_theta = np.dot(v_source_norm, v_normal_norm)
    #                     angle = math.acos(cos_theta)
    #                     to_normalize = np.cross(v_source, v_normal)
    #                     w = to_normalize/np.linalg.norm(to_normalize)
    #                     qqq = p.getQuaternionFromAxisAngle(w, angle)

    #                     qqq = Quaternion([qqq[3], qqq[0], qqq[1], qqq[2]])
    #                     qqq = qqq * Quaternion(axis=[0., 1., 0.], angle=3.14)

    #                     selectedFrame.resetPositionAndOrientation(rayInfo[0][3], [qqq[1], qqq[2], qqq[3], qqq[0]])

    #                     # (closest_points, distances, triangle_id) = objtrimesh.nearest.on_surface(np.array(rayInfo[0][3]).transpose())
    #                     # locations, index_ray, index_tri = objtrimesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)

    # for k, v in keys.items():
    #     if (k == 113) and (v == 1):
    p.stepSimulation()
    # time.sleep(1/1000) # TODO DLW
    # time.sleep(1/500)


    # count_draw = count_draw + 1

try:
    signal.pause()
except (KeyboardInterrupt, SystemExit):
    print("Shutting down...")