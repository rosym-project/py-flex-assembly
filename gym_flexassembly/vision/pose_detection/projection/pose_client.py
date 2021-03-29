import rospy

from py_flex_assembly.srv import PoseEstimation

rospy.wait_for_service('pose_estimation')
try:
    estimate = rospy.ServiceProxy('pose_estimation', PoseEstimation)
    estimation = estimate()
    print(f'Pose estimation {estimation}')
except rospy.ServiceException as e:
    print(f'Service call failed... {repr(r)}')
