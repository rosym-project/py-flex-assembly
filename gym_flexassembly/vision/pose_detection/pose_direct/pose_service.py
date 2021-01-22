#!/usr/bin/env python3

import argparse
import math
import os
import sys

# import numpy and OpenCV
import numpy as np
import cv2 as cv

import pybullet as pb
import torch


# import ros libraries
import roslib
import rospy
import message_filters
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion
from cv_bridge import CvBridge, CvBridgeError

from py_flex_assembly.srv import ClampEstimation, ClampEstimationResponse
from py_flex_assembly.msg import Clamp, ClampArray

# NOTE: this import has to come after all ROS imports, else I get the following error:
#   from cv_bridge.boost.cv_bridge_boost import getCvType',
#     'ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: cannot allocate memory in static TLS block'
from gym_flexassembly.vision.pose_detection.pose_direct import util as pose_direct_util


def create_parser():
    parser = pose_direct_util.load_model_parser(model_type='rotation')
    parser = pose_direct_util.load_model_parser(model_type='translation', parser=parser)
    return parser


class RotationEstimator():

    def __init__(self, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_rotation = pose_direct_util.load_model(args, self.device, model_type='rotation')
        self.model_rotation = self.model_rotation.eval()

    def estimate(self, img):
        input_rotation = pose_direct_util.pre_process_rotation(img) 
        return self.model_rotation(input_rotation).detach().squeeze(dim=0)

    @classmethod
    def add_args(cls, parser):
        return pose_direct_util.load_model_parser(model_type='rotation', parser=parser)


class PoseEstimator():

    def __init__(self, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_rotation = pose_direct_util.load_model(args, self.device, model_type='rotation')
        self.model_rotation = self.model_rotation.eval()
        self.model_translation = pose_direct_util.load_model(args, self.device, model_type='translation')
        self.model_translation = self.model_translation.eval()

    def estimate(self, img):
        input_rotation = pose_direct_util.pre_process_rotation(img) 
        rotation = self.model_rotation(input_rotation).detach().squeeze(dim=0)

        input_translation = pose_direct_util.pre_process_translation(img) 
        translation = self.model_translation(input_translation).detach().squeeze(dim=0)
        return {'pos': translation,
                'orn': rotation}


class PoseServer:

    def __init__(self, args, pose_estimator):
        self.args = args
        self.pose_estimator = pose_estimator

        self.bridge = CvBridge()

        self.service = rospy.Service('pose_estimations', ClampEstimation, self.handle_request)

    def handle_request(self, request):
        img = None
        # convert the raw image data
        try:
            img = self.bridge.imgmsg_to_cv2(req.color_img, "passthrough")
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        except CvBridgeError as e:
            print(e)

        
        # call the pose estimation
        pose_estimations, classes = self.pose_estimator.process_frame(depth_img, color_img)

        clamp_array = self.construct_clamp_array(pose_estimations, classes)
        if self.args.verbose:
            print("sending reply")
        return ClampEstimationResponse(clamp_array)

    def construct_clamp_array(self, pose_estimations, classes):
        # initialize the clamp array
        clamp_array = ClampArray()
        clamp_array.clamps = []
        header = Header()
        header.stamp = rospy.Time.now()
        clamp_array.header = header
        clamp_array.header.frame_id = "clamp_array"

        # add the clamps to the array
        for i in range(len(pose_estimations)):
            clamp = Clamp()
            pose = Pose()
            pose.position = Point(*pose_estimations[i][0], 0)

            # equivalent to math.acos(np.dot(np.array([1, 0]), pose_estimations[i][2]))
            yaw = math.acos(pose_estimations[i][2][0])
            # check whether the front of the clamp is shown
            roll = 0 if pose_estimations[i][3] else math.pi
            pose.orientation = Quaternion(*R.from_euler('xz', [roll, yaw]).as_quat())
            clamp.clamp_pose = pose

            if classes[i] == "large_thick_gray":
                clamp.type = Clamp.LARGE_THICK_GRAY
            elif classes[i] == "medium_thin_gray":
                clamp.type = Clamp.MEDIUM_THIN_GRAY
            elif classes[i] == "medium_thin_green":
                clamp.type = Clamp.MEDIUM_THIN_GREEN
            elif classes[i] == "small_thin_blue":
                clamp.type = Clamp.SMALL_THIN_BLUE
            else:
                print("Error: unknown clamp class \"" + classes[i] + "\"")
                rospy.signal_shutdown("an error occurred")

            clamp_array.clamps.append(clamp)
        return clamp_array


if __name__ == '__main__':
    parser = create_parser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--image', type=str, required=True,
                        help='the image file to be proessed')
    parser.add_argument('--data', type=str,
                        help='the data csv file (if available)')
    args = parser.parse_args()
    print(args)

    if args.data:
        import csv

        data = {}
        with open(args.data, 'r') as f:
            dict_reader = csv.DictReader(f)

            for row in dict_reader:
                if row['image_name'] != os.path.basename(args.image):
                    continue

                data['translation'] = [float(row['x']), float(row['y']), float(row['z'])]
                if 'qx' in row:
                    data['rotation'] = [float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])]
                else:
                    euler = [row['roll'], row['pitch'], row['yaw']]
                    euler = list(map(lambda x: float(x), euler))
                    data['rotation'] = pb.getQuaternionFromEuler(euler)


    pose_estimator = PoseEstimator(args)

    img = cv.imread(args.image, cv.IMREAD_COLOR)
    pose = pose_estimator.estimate(img)

    pos = pose['pos']
    orn = pose['orn']
    orn = orn / torch.norm(orn)

    translation_loss = torch.nn.MSELoss()
    rotation_loss = pose_direct_util.QuatLossModule()

    _pos = torch.tensor(data['translation'])
    _orn = torch.tensor(data['rotation'])

    print(f'Translation loss {translation_loss(pos, _pos):.6f}')
    print(f'Rotation loss    {rotation_loss(orn.unsqueeze(0), _orn.unsqueeze(0)):.6f}')
    print()

    orn, _orn = orn.detach().numpy(), _orn.detach().numpy()
    pos, _pos = pos.detach().numpy(), _pos.detach().numpy()

    print('Translation [x, y, z]')
    print(f'Predicted {pos}')
    print(f'Expected  {_pos}')
    print(f'Diff {100 * np.linalg.norm(pos - _pos)}cm')
    print()

    orn = np.array(pb.getEulerFromQuaternion(orn))
    _orn = np.array(pb.getEulerFromQuaternion(_orn))
    print('Rotation [roll, pitch, yaw]')
    print(f'Predicted {orn}')
    print(f'Expected  {_orn}')
    import math
    orn_diff = np.mod(np.abs(orn - _orn), 2 * math.pi) * 180 / math.pi
    print(f'Diff {max(orn_diff)}°')

    

