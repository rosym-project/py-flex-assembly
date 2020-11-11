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
from gym_flexassembly.envs.flex_assembly_env import FlexAssemblyEnv


def generate_image(model_path, model_pose, camera_settings):
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

    return rgba[:, :, :3]


width = 1280
height = 720
fov = 65
aspect = width / height
near = 0.16
far = 10
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
def get_random_camera_settings(target_pos, min_dist=0.2, max_dist=0.4, range_upper=0.3, range_lower=0.1):
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
        # flat on the ground
        # if random.randint(0, 1) == 1:
            # roll = 0
            # z = 0.71
        # else:
            # roll = math.pi
            # z = 0.73
        # pitch = 0
        # z = 0.71
    # elif base_orientation == 1:
        # long side on the ground
        # if random.randint(0, 1) == 0:
            # roll = math.pi / 2
            # z = 0.72
        # else:
            # roll = 3 * math.pi / 2
        # pitch = 0
        # z = 0.72
    # else:
        # short side on the ground
        # roll = 0
        # if random.randint(0, 1) == 0:
            # pitch = math.pi / 2
        # else:
            # pitch = 3 * math.pi / 2
        # z = 0.73

    # roll = 0
    # if random.randint(0, 1) == 1:
        # roll = math.pi
    # pitch = 0
    # z = 0.71

    if random.randint(0, 1) == 0:
        roll = math.pi / 2
        z = 0.72
    else:
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
    parser.add_argument('-v', '--visualize', action="store_true",
                        help='visualize the marker rendering and the detected markers (used for debug purposes)')
    args = parser.parse_args(args[1:])
    print(args)

    # create the output directory
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # load already existing data
    data = {}
    annotation_file = os.path.join(args.output_dir, 'data.json')
    if os.path.isfile(annotation_file):
        with open(annotation_file, mode='r') as f:
            data = json.load(f)
    else:
        data = {}

    starting_number = -1
    for i in data:
        starting_number = max(starting_number, int(i.split('.')[0]))
    starting_number += 1

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

    # collect error messages to resend them at the end
    error_msgs = ""

    # loop over the number of target images
    for j in tqdm.trange(starting_number, starting_number + args.number):
        img_name = '{:05d}'.format(j) + '.png'
        data[img_name] = []

        model_pose = generate_random_model_pose()
        camera_settings = get_random_camera_settings(model_pose['pos'])
        # print('Take image with settings:\n', camera_settings)
        # print('  pos:    ', [f'{val:.2f}' for val in camera_settings['pos']])
        # print('  target: ', [f'{val:.2f}' for val in camera_settings['target_pos']])
        # print('  up:     ', [f'{val:.2f}' for val in camera_settings['up']])

        # generate the image of the unmarked clamp and export it
        unmarked = generate_image(paths[0], model_pose, camera_settings)
        unmarked = cv.cvtColor(unmarked, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(args.output_dir, img_name), unmarked)

        # copy of the unmarked image for used to create a debug visualization
        marker_vis = np.copy(unmarked)
        #cv.imshow('Unmarked', unmarked)
        #if cv.waitKey(0) == ord('q'):
            #break

        marker_count = 0
        # loop over all markers
        for i, path in enumerate(paths[1:]):
            # generate the image of the current marker
            marker = generate_image(path, model_pose, camera_settings)

            """
            alternative mask generation
            currently fails because of small errors in the marker textures
            preferrable version for future clamps that don't include those errors

            # generate mask
            marker = cv.cvtColor(marker, cv.COLOR_RGB2BGR)
            diff = np.abs(marker - unmarked)
            mask = np.where(np.any(np.where(diff == 0, False, True), axis=2), 1, 0).astype(np.uint8)
            cv.imshow("mask", mask)
            cv.waitKey(0)
            """

            if args.visualize:
                # copy the difference between the unmarked and the marked image (i.e. the marker) to the visualization
                marker_img = cv.cvtColor(marker, cv.COLOR_RGB2BGR)
                diff = np.abs(marker_img - unmarked)
                mask = np.any(np.where(diff == 0, False, True), axis=2)
                marker_vis[mask] = marker_img[mask]

            # generate the mask
            marker = cv.cvtColor(marker, cv.COLOR_RGB2HSV)
            marker[:, :, 0] = (marker[:, :, 0] + 5) % 180
            lower = np.array([0, 245, 140])
            upper = np.array([10, 255, 255])
            mask = cv.inRange(marker, lower, upper)

            # calculate the marker position
            num_labels, _, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)

            # check how many markers were found
            if num_labels > 2:
                max_dist = 3
                # if the centroids are very close to each other,
                # it is likely that they belong to the same marker
                for i in range(1, len(centroids)):
                    are_close = False
                    # check if there is a centroid j that is close to centroid i
                    for j in range(1, len(centroids)):
                        if i == j:
                            continue
                        if np.linalg.norm(centroids[i] - centroids[j]) <= max_dist:
                            are_close = True
                            break
                    if not are_close:
                        break

                if are_close:
                    # if all centroids are close to each other,
                    # use the weighted sum of the centroids as the new centroid
                    total_area = np.sum(stats[1:, cv.CC_STAT_AREA])
                    centroids[1] *= stats[1, cv.CC_STAT_AREA] / total_area

                    for i in range(2, len(centroids)):
                        centroids[1] += centroids[i] * stats[i, cv.CC_STAT_AREA] / total_area
                else:
                    print('\nError: multiple marker detected\n')
                    print('marker path:', path)
                    print('model pose:', model_pose)
                    print('camera settings:', camera_settings)
                    error_msgs += '\nError: multiple marker detected\n'
                    error_msgs += 'marker path: ' + str(path) + '\n'
                    error_msgs += 'model pose: ' + str(model_pose) + '\n'
                    error_msgs += 'camera settings: ' + str(camera_settings) + '\n'

                    #cv.imshow("marker", cv.cvtColor(marker, cv.COLOR_HSV2BGR))
                    #cv.waitKey(0)

                    # break the marker loop and decrease j to generate a new clamp pose for the current iteration
                    j -= 1
                    break
            elif num_labels == 1:
                # only the background label was found
                continue

            marker_count += 1
            data[img_name].append({
                'id': i,
                'x': str(centroids[1].round().astype(int)[0]),
                'y': str(centroids[1].round().astype(int)[1])
            })

            if args.visualize:
                # draw a circle around the detected marker
                cv.circle(marker_vis, tuple(centroids[1].round().astype(int)), 5, [0, 255, 0], 1)

        print("\ndetected", str(marker_count), "markers")

        if args.visualize:
            cv.imshow("detected markers", marker_vis)
            cv.waitKey(0)
            cv.destroyAllWindows()

    with open(os.path.join(args.output_dir, 'data.json'), 'w') as outfile:
        json.dump(data, outfile, indent=2)

    if len(error_msgs) > 0:
        print("==============================================")
        print(" Errors that occured during marker generation")
        print("==============================================")
        print(error_msgs)

if __name__ == '__main__':
    main(sys.argv)
