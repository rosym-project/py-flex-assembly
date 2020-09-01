#!/usr/bin/env python3

import pybullet as p

import roslib
import rospy

from geometry_msgs.msg import Pose, Point, Quaternion

from py_flex_assembly.msg import Clamp, ClampArray

from py_flex_assembly.srv import ClampPose, ClampPoseResponse
from py_flex_assembly.srv import DeleteClamp, DeleteClampResponse
from py_flex_assembly.srv import SpawnClamp, SpawnClampResponse

from gym_flexassembly.smartobjects.spring_clamp import SpringClamp

class ClampService():

    def __init__(self):
        self._clamps = {}

        # rospy.init_node('clamp_service', anonymous=True)
        rospy.Service('spawn_clamp', SpawnClamp, self.spawn_clamp_ros)
        rospy.Service('delete_clamp', DeleteClamp, self.delete_clamp_ros)
        rospy.Service('clamp_pose', ClampPose, self.clamp_pose_ros)

    def spawn_clamp(self, **kwargs):
        print('Spawn clamp with kwargs: ' + str(kwargs))
        clamp = SpringClamp(**kwargs)
        print('Reset clamp...')
        clamp.reset()
        print('Save in dict')
        self._clamps[clamp.getModelId()] = clamp
        return clamp.getModelId() 

    def spawn_clamp_ros(self, request):
        # TODO: also handle variant
        print('Spawn clamp via ROS...')
        return self.spawn_clamp(pos=request.pose.position, orn=request.pose.orientation)

    def delete_clamp(self, model_id):
        if model_id not in self._clamps:
            raise ValueError('Clamp[%s] does not exist!' % model_id)

        p.removeBody(model_id)
        del self._clamps[model_id]

        return DeleteClampResponse()

    def delete_clamp_ros(self, request):
        return self.delete_clamp(request.model_id)

    def clamp_pose_ros(self, request):
        clamp_array = ClampArray()
        model_ids = []

        for model_id in self._clamps:
            pos, orn = p.getBasePositionAndOrientation(model_id)

            pose = Pose()
            pose.position = Point(*pos)
            pose.orientation = Quaternion(*orn)

            clamp = Clamp()
            clamp.clamp_pose = pose
            # TODO should be resolved from variant
            clamp.type = Clamp.LARGE_THICK_GRAY

            clamp_array.clamps.append(clamp)
            model_ids.append(model_id)

        return ClampPoseResponse(clamp_array, model_ids)
