import argparse

from scipy.spatial.transform import Rotation as R

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion

parser = argparse.ArgumentParser(description='Send a pose to a given topic to simulate the robots arm pose',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--topic', type=str, default="/robot/fdb/cart_pose_0",
                    help='the topic on which the pose is published')
parser.add_argument('-p', '--position', type=float, nargs=3, required=True,
                    help='the position of the arm as [x, y, z] in meter')
parser.add_argument('-e', '--euler', type=float, nargs=3,
                    help='the orientation of the arm as euler angles [a, b, c] in degrees')
parser.add_argument('-q', '--quat', type=float, nargs=4,
                    help='the orientation of the arm as a quaternion [qx, qy, qz, qw]')
args = parser.parse_args()

if args.euler is None and args.quat is None:
    print('You have to provide a rotation! Either a quaternion or as Euler angles.')
    exit(1)

print(args)

pos = Point()
pos.x = args.position[0]
pos.y = args.position[1]
pos.z = args.position[2]

if args.euler:
    _orn = R.from_euler('zyx', args.euler, degrees=True)
else:
    _orn = R.from_quat(args.quat)
_orn = _orn.as_quat()

orn = Quaternion()
orn.x = _orn[0]
orn.y = _orn[1]
orn.z = _orn[2]
orn.w = _orn[3]

pose = Pose()
pose.position = pos
pose.orientation = orn

print(f'Send pose:m\n{pose}')

pub = rospy.Publisher(args.topic, Pose, queue_size=1)
rospy.init_node('send_pose', anonymous=True)
pub.publish(pose)
