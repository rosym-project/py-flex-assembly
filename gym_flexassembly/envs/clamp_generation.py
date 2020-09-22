import argparse
import json
import math
import random
import os
import sys
import time

import cv2 as cv
import numpy as np
import pybullet as p
import pybullet_data
import tqdm

from gym_flexassembly import data as flexassembly_data


def generate_image(model_settings, camera_settings):
    # connect to the physics simulation
    physicsClient = p.connect(p.DIRECT)

    # place the ground plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF('plane.urdf')

    # place a clamp
    clamp_id = p.loadSDF(model_settings['path'], globalScaling=2.0)[0]
    p.resetBasePositionAndOrientation(clamp_id, model_settings['pos'], model_settings['rot'])

    # set up the camera
    width = 1280
    height = 720

    view_matrix = p.computeViewMatrix(camera_settings['pos'],
                                      camera_settings['target_pos'],
                                      camera_settings['up'])

    fov = 65
    aspect = width / height
    near = 0.16
    far = 10
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # take a picture
    w, h, rgba, depth_buffer, segmentation = p.getCameraImage(width,
                              height,
                              view_matrix,
                              projection_matrix,
                              shadow=True,
                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # stop the physics simulation
    p.disconnect()
    return rgba[:, :, :3]


def get_random_camera_settings(min_dist=0.2, max_dist=0.5, max_vert_up=0.3, max_ver_low=0.1):
    pos_x = random.uniform(-max_vert_up, max_vert_up)
    pos_y = random.uniform(-max_vert_up, max_vert_up)
    pos_z = random.uniform(min_dist, max_dist)
    pos = [pos_x, pos_y, pos_z]

    target_x = random.uniform(-max_ver_low, max_ver_low)
    target_y = random.uniform(-max_ver_low, max_ver_low)
    target_z = 0 
    target_pos = [target_x, target_y, target_z]

    up_x = random.uniform(-1, 1)
    up_y = random.uniform(-1, 1)
    up_z = random.uniform(-0.1, 0.1)
    up = [up_x, up_y, up_z]

    return {'pos': pos,
            'target_pos': target_pos,
            'up': up}


def get_model_settings(path, pos, rot):
    return {'path': path,
            'pos': pos,
            'rot': rot}


def main(args):
    parser = argparse.ArgumentParser('Generate images and annotations for pose detection of clamps',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', type=str, default='./output/',
                        help='the directory in which the resulting images and annotations are saved')
    parser.add_argument('-n', '--number', type=int, default=10,
                        help='the number of images to be generated')
    parser.add_argument('--starting_number', type=int, default=0,
                        help='the number with which image generation starts')
    parser.add_argument('-c', '--clamp_dir', type=str, default='objects/marked_clamps/clamp_1',
                        help='the directory of the clamp of which data is generated relative to the flex assembly data dir')
    args = parser.parse_args(args[1:])
    print(args)

    # create the output directory
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # create the json object that stores the marker positions
    data = {}

    # load the csv file containing the marker points and the paths
    clamp_dir = os.path.join(flexassembly_data.getDataPath(), args.clamp_dir)
    marker_file = os.path.join(clamp_dir, 'marker.csv')
    paths = np.loadtxt(marker_file, delimiter=',', skiprows=1, usecols=[0], dtype=str)
    paths = [os.path.join(clamp_dir, p) for p in paths]
    marker = np.loadtxt(marker_file, delimiter=',', skiprows=2, usecols=[2, 3, 4])

    for j in tqdm.trange(args.starting_number, args.number):
        img_name = '{:05d}'.format(j) + '.png'
        data[img_name] = []

        translation = [0, 0, 0]
        if random.random() > 0.5:
            x_rot = math.pi
        else:
            x_rot = 0
        rotation = p.getQuaternionFromEuler([x_rot, 0, random.uniform(0, 2 * math.pi)])

        camera_settings = get_random_camera_settings()
        # print('Take image with settings:\n', camera_settings)
        # print('  pos:    ', [f'{val:.2f}' for val in camera_settings['pos']])
        # print('  target: ', [f'{val:.2f}' for val in camera_settings['target_pos']])
        # print('  up:     ', [f'{val:.2f}' for val in camera_settings['up']])

        # generate the image of the unmarked clamp
        unmarked = generate_image(get_model_settings(paths[0], translation, rotation), camera_settings)
        unmarked = cv.cvtColor(unmarked, cv.COLOR_RGB2BGR)
        # cv.imshow('Unmarked', unmarked)
        # cv.waitKey(0)
        cv.imwrite(os.path.join(args.output_dir, img_name), unmarked)

        for i, path in enumerate(paths[1:]):
            # generate the image of the current marker
            marker = generate_image(get_model_settings(path, translation, rotation), camera_settings)

            # generate the mask
            marker = cv.cvtColor(marker, cv.COLOR_RGB2HSV)
            marker[:, :, 0] = (marker[:, :, 0] + 5) % 180
            lower = np.array([0, 200, 200])
            upper = np.array([10, 255, 255])
            mask = cv.inRange(marker, lower, upper)

            # calculate the marker position
            num_labels, _, _, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 2:
                print('\nError: multiple marker detected\n')
                exit()
            elif num_labels == 1:
                # only the background label was found
                continue

            data[img_name].append({
                'id': i,
                'x': str(centroids[1].round().astype(int)[0]),
                'y': str(centroids[1].round().astype(int)[1])
            })

            # draw the marker onto the image
            # cv.circle(unmarked, tuple(centroids[1].round().astype(int)), 2, [0, 255, 0], -1)

    with open(os.path.join(args.output_dir, 'data.json'), 'a') as outfile:
        json.dump(data, outfile, indent=2)

        # cv.imshow('img', unmarked)
        # cv.waitKey(0)

if __name__ == '__main__':
    main(sys.argv)
