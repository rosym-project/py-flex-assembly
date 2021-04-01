import argparse
import math
import random
import os
import sys
import time

import rospy
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p
import pybullet_data

import PIL
import torch
import torchvision

from gym_flexassembly import data as flexassembly_data
from gym_flexassembly.envs.flex_assembly_env import FlexAssemblyEnv
from gym_flexassembly.vision.pose_detection.pose_direct import models, datasets

import gym_flexassembly.vision.util_dataset as util_dataset
import gym_flexassembly.vision.pose_detection.pose_direct.pose_service as pose_service

parser = pose_service.create_parser()
args = parser.parse_args()
print(args)
pose_estimator = pose_service.PoseEstimator(args)


# load the csv file containing the marker points and the paths
clamp_dir = os.path.join(flexassembly_data.getDataPath(), 'objects/marked_clamps/clamp_1')
marker_file = os.path.join(clamp_dir, 'marker.csv')
paths = np.loadtxt(marker_file, delimiter=',', skiprows=1, usecols=[0], dtype=str)
paths = [os.path.join(clamp_dir, p) for p in paths]

util_dataset.setup_env()

model_pose = util_dataset.generate_random_model_pose()
camera_settings = util_dataset.get_random_camera_settings(model_pose['pos'])

background = util_dataset.get_image(camera_settings)

clamp_id = p.loadSDF(os.path.join(clamp_dir, 'unmarked/clamp_1_unmarked.sdf'))[0]
p.resetBasePositionAndOrientation(clamp_id, model_pose['pos'], model_pose['orn'])


view_matrix = np.array(camera_settings['view_matrix']).reshape((4, 4)).transpose()

rgb = util_dataset.get_image(camera_settings)


img = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

print('Estimate pose...')
since = time.time()
pose = pose_estimator.estimate(img)
diff = time.time() - since
print(f'Pose estimation took {diff:.3f}s')

pos, orn = util_dataset.from_camera_to_global(pose['pos'], pose['orn'], view_matrix)

print(orn)
print(model_pose['orn'])
coordinate_system_id = p.loadURDF(os.path.join(flexassembly_data.getDataPath(), 'objects', 'coordinate_system.urdf'))
p.resetBasePositionAndOrientation(coordinate_system_id, model_pose['pos'], model_pose['orn'])

coordinate_system_id = p.loadURDF(os.path.join(flexassembly_data.getDataPath(), 'objects', 'coordinate_system.urdf'))
p.resetBasePositionAndOrientation(coordinate_system_id, pos, orn)

def format_number_list(l):
    return list(map(lambda n: f'{n:.4f}', l))

rot_euler = np.array(p.getEulerFromQuaternion(orn))
rot_euler_m = np.array(p.getEulerFromQuaternion(model_pose['orn']))
rot_diff = np.mod(np.abs(rot_euler - rot_euler_m), 2 * math.pi) * 180 / math.pi

print('Translation')
print(format_number_list(pos))
print(format_number_list(model_pose['pos']))
print('Diff:', np.linalg.norm(np.array(pos) - np.array(model_pose['pos'])))
print()
print('Rotation')
print(format_number_list(rot_euler))
print(format_number_list(rot_euler_m))
print('Diff:', rot_diff)

rospy.spin()
