import time

import pyquaternion
import rospy

from cosima_msgs.srv import Move, MoveRequest 
from std_srvs.srv import Empty

rospy.init_node('move_arm_example', anonymous=False)

# Code to move the arm
# wait until ROS server for arm movement is available and save as function
rospy.wait_for_service('/css/move_srv')
move = rospy.ServiceProxy('/css/move_srv', Move)

# Init request
move_req = MoveRequest()
# init position
move_req.i_pose.position.x = -0.0452333
move_req.i_pose.position.y = -0.594722
move_req.i_pose.position.z = 0.301398
# init orientation
orn = pyquaternion.Quaternion(w=-0.0351002,x=-0.353731,y=0.934325,z=-0.0260594) * pyquaternion.Quaternion(axis=[0, 0, 1], angle=0.0 / 180.0 * 3.14159265)
move_req.i_pose.orientation.x = orn[1]
move_req.i_pose.orientation.y = orn[2]
move_req.i_pose.orientation.z = orn[3]
move_req.i_pose.orientation.w = orn[0]
# set movement speed (these are frequencies; thus lower values mean faster)
move_req.i_max_trans_sec = 30.0
move_req.i_max_rot_sec = 50.0
# call service
move(move_req)


# Code to open the gripper
# rospy.wait_for_service('/gripper1/open_gripper')
# open_gripper = rospy.ServiceProxy('/gripper1/open_gripper', Empty)
# open_gripper()
# time.sleep(2.0)

# Code to close the gripper
# rospy.wait_for_service('/gripper1/close_gripper')
# close_gripper = rospy.ServiceProxy('/gripper1/close_gripper', Empty)
# close_gripper()
# time.sleep(2.0)


