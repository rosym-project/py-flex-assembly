#!/usr/bin/env python3

import roslib
import rospy

from py_flex_assembly.srv import SpawnClamp, SpawnClampResponse
from py_flex_assembly.srv import DeleteClamp, DeleteClampResponse
from py_flex_assembly.srv import ClampPose, ClampPoseResponse

from geometry_msgs.msg import Pose, Point, Quaternion

rospy.wait_for_service('clamp_pose')
server = rospy.ServiceProxy('clamp_pose', ClampPose)
response = server()

print(response)
print()
print('=========================================')
print()

rospy.wait_for_service('spawn_clamp')
server = rospy.ServiceProxy('spawn_clamp', SpawnClamp)

pose = Pose()
pose.position = Point(0.15, 0.2, 0.75)
pose.orientation = Quaternion(0, 0, 1, 0)
variant = "test_variant"
response = server(pose, variant)

print(response)
print()
print('=========================================')
print()

# rospy.wait_for_service('delete_clamp')
# server = rospy.ServiceProxy('delete_clamp', DeleteClamp)
# response = server(4)

# print(response)
# print()
# print('=========================================')
# print()
