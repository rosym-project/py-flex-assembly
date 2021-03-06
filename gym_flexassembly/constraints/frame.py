import numpy as np
import math
import os

import gym_flexassembly.data as data

import rospy
from geometry_msgs.msg import TransformStamped

import pyquaternion

class Frame(object):
    """ TODO
    """

    def __init__(self, pb, text, fixed_base=True, ref_name="world", ref_id=-1, ref_link_id=-1, is_body_frame=False, scale=-1, dirtext=[0,0.15,0.15]):
        self.p = pb
        self.text = text
        self.ref_id = ref_id
        self.ref_link_id = ref_link_id
        self.ref_name = ref_name
        self.is_body_frame = is_body_frame
        self.dirtext = dirtext
        urdfRootPath = data.getDataPath()

        # calculate scaling
        scaling = 1.0
        # Get one time the initial axisaligned bounding box to calculate the size of the frame
        if is_body_frame and scale <= 0:
            aabbMin, aabbMax = self.p.getAABB(self.ref_id, self.ref_link_id)
            diff_x = math.fabs(aabbMax[0] - aabbMin[0])
            diff_y = math.fabs(aabbMax[1] - aabbMin[1])
            diff_z = math.fabs(aabbMax[2] - aabbMin[2])
            mid = (((diff_x + diff_y) * 0.5) + diff_z) * 0.5
            scaling = min(max(mid,0.1),0.5)
        elif scale > 0:
            scaling = scale

        self.frame_ghost_id = self.p.loadURDF(os.path.join(urdfRootPath, "visualization/frame_full.urdf"), useFixedBase=fixed_base, globalScaling=scaling)
        self.p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.frame_ghost_id, parentLinkIndex=-1)
        self.p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.frame_ghost_id, parentLinkIndex=-1)
        self.p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.frame_ghost_id, parentLinkIndex=-1)
        self.frame_text_node = self.p.addUserDebugText(str(self.text), self.dirtext,
                        textColorRGB=[0, 0, 0],
                        textSize=1.0,
                        parentObjectUniqueId=self.frame_ghost_id,
                        parentLinkIndex=-1)
        self.p.addUserDebugLine([0, 0.05, 0.05], np.array(self.dirtext)*0.9, [0, 0, 0], parentObjectUniqueId=self.frame_ghost_id, parentLinkIndex=-1)

        self.deselectedTransparency = 0.5
        self.selectedTransparency = 1
        self.transparency = self.deselectedTransparency
        self.visibility = [True, True, True, True, True, True]

        self.updateVisual()

        # Add collision disabling stuff! Binary, ray only works on 0x1
        collisionFilterGroup = 0x100
        collisionFilterMask  = 0x1
        for i in range(self.p.getNumJoints(self.frame_ghost_id)):
            self.p.setCollisionFilterGroupMask(self.frame_ghost_id, i-1, collisionFilterGroup, collisionFilterMask)

        # Conclusion on p.setCollisionFilterGroupMask(id, (body_id) -1, collisionFilterGroup, collisionFilterMask):
        # Raytest works with collisionFilterMask = 0x1 ONLY
        # EVERYTHING is BINARY
        # DIFFERENT GROUP IDs == NO COLLISION!
        # However, groups need to be assigned at the beginning!

        # enableCollision = 1
        # p.setCollisionFilterPair(planeId, cubeId, -1, -1, enableCollision)

        self.internal_pos = [0,0,0]
        self.internal_orn = [0,0,0,1]

        self.absolute_pos = self.internal_pos
        self.absolute_orn = self.internal_orn

        self.t = TransformStamped()
        self.t.header.stamp = rospy.Time.now()
        self.t.header.frame_id = self.ref_name
        # USE INTERNAL ID AND MAKE UNIQUE?
        self.t.child_frame_id = self.text
        self.t.transform.translation.x = self.internal_pos[0]
        self.t.transform.translation.y = self.internal_pos[1]
        self.t.transform.translation.z = self.internal_pos[2]
        self.t.transform.rotation.x = self.internal_orn[0]
        self.t.transform.rotation.y = self.internal_orn[1]
        self.t.transform.rotation.z = self.internal_orn[2]
        self.t.transform.rotation.w = self.internal_orn[3]

    def getROSTransformStamped(self):
        return self.t

    def updateVisual(self):
        self.p.changeVisualShape(self.frame_ghost_id, 0, rgbaColor=[1, 0, 0, (self.transparency if self.visibility[0] else 0)]) # X
        self.p.changeVisualShape(self.frame_ghost_id, 2, rgbaColor=[0, 1, 0, (self.transparency if self.visibility[1] else 0)]) # Y
        self.p.changeVisualShape(self.frame_ghost_id, 4, rgbaColor=[0, 0, 1, (self.transparency if self.visibility[2] else 0)]) # Z
        self.p.changeVisualShape(self.frame_ghost_id, 1, rgbaColor=[1, 0, 0, (self.transparency if self.visibility[3] else 0)]) # RotX
        self.p.changeVisualShape(self.frame_ghost_id, 3, rgbaColor=[0, 1, 0, (self.transparency if self.visibility[4] else 0)]) # RotY
        self.p.changeVisualShape(self.frame_ghost_id, 5, rgbaColor=[0, 0, 1, (self.transparency if self.visibility[5] else 0)]) # RotZ

    def select(self):
        self.transparency = self.selectedTransparency
        self.updateVisual()

    def deselect(self):
        self.transparency = self.deselectedTransparency
        self.updateVisual()

    def setVisibility(self, index, visible):
        if index == 0:
            self.visibility[index] = visible
            self.p.changeVisualShape(self.frame_ghost_id, 0, rgbaColor=[1, 0, 0, (self.transparency if visible else 0)]) # X
        elif index == 1:
            self.visibility[index] = visible
            self.p.changeVisualShape(self.frame_ghost_id, 2, rgbaColor=[0, 1, 0, (self.transparency if visible else 0)]) # Y
        elif index == 2:
            self.visibility[index] = visible
            self.p.changeVisualShape(self.frame_ghost_id, 4, rgbaColor=[0, 0, 1, (self.transparency if visible else 0)]) # Z
        elif index == 3:
            self.visibility[index] = visible
            self.p.changeVisualShape(self.frame_ghost_id, 1, rgbaColor=[1, 0, 0, (self.transparency if visible else 0)]) # RotX
        elif index == 4:
            self.visibility[index] = visible
            self.p.changeVisualShape(self.frame_ghost_id, 3, rgbaColor=[0, 1, 0, (self.transparency if visible else 0)]) # RotY
        elif index == 5:
            self.visibility[index] = visible
            self.p.changeVisualShape(self.frame_ghost_id, 5, rgbaColor=[0, 0, 1, (self.transparency if visible else 0)]) # RotZ

    def getFrameId(self):
        return self.frame_ghost_id

    def setFrameText(self, text):
        self.text = text
        self.frame_text_node = self.p.addUserDebugText(str(self.text), self.dirtext,
                        textColorRGB = [0, 0, 0],
                        textSize = 1.0,
                        parentObjectUniqueId = self.frame_ghost_id,
                        parentLinkIndex = -1,
                        replaceItemUniqueId = self.frame_text_node)

    def resetPositionAndOrientation(self, pos, orn):
        self.internal_pos = pos
        self.internal_orn = orn

        if self.is_body_frame:
            if self.ref_link_id > -1:
                # TODO check indexing
                # (x, y, z), (a, b, c, d), _, _, _, _
                # linkWorldPosition, linkWorldOrientation, localInertialFramePosition, localInertialFrameOrientation, worldLinkFramePosition, worldLinkFrameOrientation = self.p.getLinkState(self.ref_link_id, self.ref_id)
                _, _, _, _, worldLinkFramePosition, worldLinkFrameOrientation = self.p.getLinkState(self.ref_id, self.ref_link_id)
                self.absolute_pos = worldLinkFramePosition

                pp = pyquaternion.Quaternion(w=worldLinkFrameOrientation[3],x=worldLinkFrameOrientation[0],y=worldLinkFrameOrientation[1],z=worldLinkFrameOrientation[2]) * pyquaternion.Quaternion(w=self.internal_orn[3],x=self.internal_orn[0],y=self.internal_orn[1],z=self.internal_orn[2])
                # self.absolute_orn = worldLinkFrameOrientation * self.internal_orn
                self.absolute_orn = [pp.x, pp.y, pp.z, pp.w]
                self.p.resetBasePositionAndOrientation(self.frame_ghost_id, self.absolute_pos, self.absolute_orn)
            else:
                rpos, rorn = self.p.getBasePositionAndOrientation(self.ref_id)
                self.absolute_pos = rpos
                # self.absolute_orn = rorn

                pp = pyquaternion.Quaternion(w=rorn[3],x=rorn[0],y=rorn[1],z=rorn[2]) * pyquaternion.Quaternion(w=self.internal_orn[3],x=self.internal_orn[0],y=self.internal_orn[1],z=self.internal_orn[2])
                self.absolute_orn = [pp.x, pp.y, pp.z, pp.w]
                self.p.resetBasePositionAndOrientation(self.frame_ghost_id, self.absolute_pos, self.absolute_orn)
        else:
            if self.ref_id > -1:
                rpos, rorn = self.p.getBasePositionAndOrientation(self.ref_id)
                npos, norn = self.p.multiplyTransforms(rpos,rorn,pos,orn)
                self.absolute_pos = npos
                self.absolute_orn = norn
                self.p.resetBasePositionAndOrientation(self.frame_ghost_id, npos, norn)
            else:
                self.p.resetBasePositionAndOrientation(self.frame_ghost_id, pos, orn)
                self.absolute_pos = pos
                self.absolute_orn = orn

        self.t.transform.translation.x = self.internal_pos[0]
        self.t.transform.translation.y = self.internal_pos[1]
        self.t.transform.translation.z = self.internal_pos[2]
        self.t.transform.rotation.x = self.internal_orn[0]
        self.t.transform.rotation.y = self.internal_orn[1]
        self.t.transform.rotation.z = self.internal_orn[2]
        self.t.transform.rotation.w = self.internal_orn[3]

    def getInternalPosOrn(self):
        return self.internal_pos, self.internal_orn

    def getAbsolutePosOrn(self):
        return self.absolute_pos, self.absolute_orn

    def getRefId(self):
        return self.ref_id

    def getName(self):
        return self.text

    def isBodyFrame(self):
        return self.is_body_frame
