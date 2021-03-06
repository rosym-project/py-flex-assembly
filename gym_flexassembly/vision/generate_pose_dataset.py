import argparse
import csv
import math
import random
import os
import sys
import time

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p
import pybullet_data
import tqdm

from gym_flexassembly import data as flexassembly_data
from gym_flexassembly.envs.flex_assembly_env import FlexAssemblyEnv

def drawAxis(img, model_pose, rotation_matrix, translation_vec):
    """
    Draw the axis of the model pose on an image as seen through a
    camera with the provided rotation matrix and translation_vector.
    """
    # guess camera matrix and distortion coefficients
    focal_length = img.shape[1]
    center = (img.shape[1] / 2, img.shape[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype='double')
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # compute rodrigues vector from rotation matrix
    rotation_vec, _ = cv.Rodrigues(rotation_matrix)

    # construct axis for model pose
    origin = np.array(model_pose['pos'])
    _rotation_matrix = p.getMatrixFromQuaternion(model_pose['orn'])
    _rotation_matrix = np.array(_rotation_matrix).reshape(3, 3)
    x = origin + _rotation_matrix.dot(np.array([0.1, 0, 0]))
    y = origin + _rotation_matrix.dot(np.array([0, 0.1, 0]))
    z = origin + _rotation_matrix.dot(np.array([0, 0, 0.1]))
    axis = np.array([origin, x, y, z])

    # project axis
    imgpts, _ = cv.projectPoints(axis, rotation_vec, translation_vec, camera_matrix, dist_coeffs)

    # extract projected points as 2d int tuples
    _origin = tuple(imgpts[0].astype(np.int32).ravel())
    _x = tuple(imgpts[1].astype(np.int32).ravel())
    _y = tuple(imgpts[2].astype(np.int32).ravel())
    _z = tuple(imgpts[3].astype(np.int32).ravel())

    # draw lines for axis
    img = cv.line(img, _origin, _x, (255, 0, 0), 3)
    img = cv.line(img, _origin, _y, (0, 255, 0), 3)
    img = cv.line(img, _origin, _z, (0, 0, 255), 3)
    return img


def generate_image(model_path, model_pose, camera_settings):
    # take a picture of the background
    w, h, background, depth_buffer, segmentation = p.getCameraImage(
                              camera_settings['width'],
                              camera_settings['height'],
                              camera_settings['view_matrix'],
                              camera_settings['projection_matrix'],
                              shadow=True,
                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # place a clamp
    clamp_id = p.loadSDF(model_path)[0]
    p.resetBasePositionAndOrientation(clamp_id, model_pose['pos'], model_pose['orn'])

    # take a picture
    w, h, rgba, depth_buffer, segmentation = p.getCameraImage(
                              camera_settings['width'],
                              camera_settings['height'],
                              camera_settings['view_matrix'],
                              camera_settings['projection_matrix'],
                              shadow=True,
                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # remove model
    p.removeBody(clamp_id)

    return background[:, :, :3], rgba[:, :, :3]


width = 1280
height = 720
fov = 65
aspect = width / height
near = 0.16
far = 10
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
def get_random_camera_settings(target_pos, min_dist=0.2, max_dist=0.25, range_upper=0.05, range_lower=0.01):
    pos_x = target_pos[0] + random.uniform(-range_upper, range_upper)
    pos_y = target_pos[1] + random.uniform(-range_upper, range_upper)
    pos_z = target_pos[2] + random.uniform(min_dist, max_dist)
    pos = [pos_x, pos_y, pos_z]

    target_x = target_pos[0] + random.uniform(-range_lower, range_lower)
    target_y = target_pos[1] + random.uniform(-range_lower, range_lower)
    target_z = target_pos[2]
    target_pos = [target_x, target_y, target_z]

    up_x = random.uniform(-1, 1)
    up_y = random.uniform(-1, 1)
    up_z = random.uniform(-0.1, 0.1)
    up = [up_x, up_y, up_z]

    view_matrix = p.computeViewMatrix(pos,
                                      target_pos,
                                      up)
    return {'pos': pos,
            'target_pos': target_pos,
            'up': up,
            'view_matrix': view_matrix,
            'projection_matrix': projection_matrix,
            'width': width,
            'height': height}


table_offset_x = 0.1
table_offset_y = 1.2
def generate_random_model_pose():
    # orientation
    # always rotate around z
    # normal orn: [0, 0, x]
    # normal height: 0.71

    # low prob orn: [math.pi / 2, 0, x] or [3 * math.pi, 0, x]
    # height: 0.73

    # lying
    # orn [0, math.pi / 2, x] or [0, 3 * math.pi, x]
    # height: 0.705
    # base_orientation = random.randint(0, 2)
    # if base_orientation == 0:
        # # flat on the ground
        # if random.randint(0, 1) == 1:
            # roll = 0
            # z = 0.71
        # else:
            # roll = math.pi
            # z = 0.73
        # pitch = 0
        # z = 0.71
    # elif base_orientation == 1:
        # # long side on the ground
        # if random.randint(0, 1) == 0:
            # roll = math.pi / 2
            # z = 0.72
        # else:
            # roll = 3 * math.pi / 2
        # pitch = 0
        # z = 0.72
    # else:
        # # short side on the ground
        # roll = 0
        # if random.randint(0, 1) == 0:
            # pitch = math.pi / 2
        # else:
            # pitch = 3 * math.pi / 2
        # z = 0.73

    roll = 3 * math.pi / 2
    pitch = 0
    z = 0.72

    yaw = random.uniform(0, 2 * math.pi)
    rotation = p.getQuaternionFromEuler([roll, pitch, yaw])

    # table top goes from x[-1.07, -0.01] y[-0.03, -1.05]
    border_size = 0.25
    table_range_x = [-1.07 + border_size, -0.01 - border_size]
    table_range_y = [-1.05 + border_size, -0.03 - border_size]

    x = random.uniform(*table_range_x) + table_offset_x
    y = random.uniform(*table_range_y) + table_offset_y
    position = [x, y, z]

    return {'pos': position, 'orn': rotation}



def main(args):
    parser = argparse.ArgumentParser('Generate images and annotations for pose detection of clamps',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', type=str, default='./output/',
                        help='the directory in which the resulting images and annotations are saved')
    parser.add_argument('-n', '--number', type=int, default=10,
                        help='the number of images to be generated')
    parser.add_argument('-c', '--clamp_dir', type=str, default='objects/marked_clamps/clamp_1',
                        help='the directory of the clamp of which data is generated relative to the flex assembly data dir')
    parser.add_argument('--crop_image', action="store_true",
                        help='crop the image centered on the clamp')
    parser.add_argument('--aspect_ratio', type=float, default=1280/720,
                        help='the aspect ratio of the cropped image')
    parser.add_argument('--border_ratio', type=float, default=0.2,
                        help='the percentage of the image size that is added as a border around the crop')
    args = parser.parse_args(args[1:])
    print(args)

    # create the output directory
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    annotation_file = os.path.join(args.output_dir, 'data.csv')
    starting_number = 0
    if os.path.isfile(annotation_file):
        with open(annotation_file, mode='r') as f:
            starting_number = max(0, len(f.readlines()) - 1)
            print('Extracted starting number..', starting_number)
    else:
        #header = ["id", "image_name", "camera_qx", "camera_qy", "camera_qz", "camera_qw", "x", "y", "z", "roll", "pitch", "yaw"]
        header = ["id", "image_name", "camera_qx", "camera_qy", "camera_qz", "camera_qw", "x", "y", "z", "qx", "qy", "qz", "qw"]
        with open(annotation_file, mode='w') as f:
            csv.writer(f, delimiter=',').writerow(header)

    # load the csv file containing the marker points and the paths
    clamp_dir = os.path.join(flexassembly_data.getDataPath(), args.clamp_dir)
    marker_file = os.path.join(clamp_dir, 'marker.csv')
    paths = np.loadtxt(marker_file, delimiter=',', skiprows=1, usecols=[0], dtype=str)
    paths = [os.path.join(clamp_dir, p) for p in paths]

    # set up the simulation environment
    env = FlexAssemblyEnv(stepping=False, gui=False)
    env.remove_camera('global')
    for clamp_id in env.object_ids['clamps']:
        p.removeBody(clamp_id)
    for coordinate_id in env.object_ids['coordinate_systems']:
        p.removeBody(coordinate_id)

    # loop over the number of target images
    j = starting_number
    while (j < args.number + starting_number):
        data = np.empty(13, dtype=object)
        img_name = '{:05d}'.format(j) + '.png'
        data[0] = j
        data[1] = img_name

        model_pose = generate_random_model_pose()
        camera_settings = get_random_camera_settings(model_pose['pos'])

        # extract the camera rotation from the camera settings
        view_matrix = np.array(camera_settings['view_matrix']).reshape((4, 4)).transpose()
        camera_rot = np.linalg.inv(view_matrix[:3, :3])
        data[2:6] = R.from_matrix(camera_rot).as_quat()

        # transform the pose from global coordinates to camera coordinates
        t = view_matrix.dot(np.array([*model_pose['pos'], 1]))[:3]
        rot = np.array(p.getMatrixFromQuaternion(model_pose['orn'])).reshape((3, 3))
        rot = view_matrix[:3, :3].dot(rot)
        data[6:9] = t
        #data[9:12] = p.getEulerFromQuaternion(R.from_matrix(rot).as_quat())
        #data[9:12] = p.getEulerFromQuaternion(R.from_matrix(rot).as_quat())
        data[9:13] = R.from_matrix(rot).as_quat()

        # generate the image of the clamp and the background
        background, clamp_img = generate_image(paths[0], model_pose, camera_settings)

        # View debug drawing of axis
        # res = cv.cvtColor(clamp_img, cv.COLOR_RGB2BGR)
        # res = drawAxis(res, model_pose, view_matrix[:3, :3], view_matrix[:3, 3])

        # cv.imshow('Axis Projection', res)
        # cv.waitKey(0)
        # exit()

        if args.crop_image:
            # extract the clamp area
            diff = background - clamp_img
            clamp_mask = np.where(np.any(np.where(diff == 0, False, True), axis=2), 1, 0).astype(np.uint8)
            num_labels, _, stats, centroids = cv.connectedComponentsWithStats(clamp_mask, connectivity=8)

            if num_labels != 2:
                # the clamp is outside of the image or multiple clamps are visible
                # don't save the image and rerun the iteration
                print('\033[93m' + str(num_labels - 1), "clamps detected. Rerunning iteration" + '\033[0m')
                continue

            width = stats[1, cv.CC_STAT_WIDTH]
            height = stats[1, cv.CC_STAT_HEIGHT]
            top = stats[1, cv.CC_STAT_TOP]
            left = stats[1, cv.CC_STAT_LEFT]
            bottom = top + height
            right = left + width

            # the following calculations don't take edge cases into account,
            # since we know that the clamp is in the middle of the image
            if width / height < args.aspect_ratio:
                # add border in y direction
                top = top - round(height * args.border_ratio * 0.5)
                height = round(height * (args.border_ratio + 1))
                # adapt the width according to the aspect ratio
                width = round(args.aspect_ratio * height)
                left = round(centroids[1, 0] - width / 2)
            else:
                # add border in x direction
                left = left - round(width * args.border_ratio * 0.5)
                width = round(width * (args.border_ratio + 1))
                # adapt the height according to the aspect ratio
                height = round(width / args.aspect_ratio)
                top = round(centroids[1, 1] - height / 2)

            # execute the crop
            clamp_img = clamp_img[top : top + height, left : left + width]

        if clamp_img is None:
            print(f'Could not extract ROI of image! {camera_settings}, {model_pose}. Rerun iteration.')
            continue

        with open(annotation_file, mode='a') as f:
            csv.writer(f, delimiter=',').writerow(list(data))

        # export the image
        clamp_img = cv.cvtColor(clamp_img, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(args.output_dir, img_name), clamp_img)

        j += 1

if __name__ == '__main__':
    main(sys.argv)
