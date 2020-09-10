import sys
import os
import time
import json

import cv2 as cv
import numpy as np
import pybullet as p
import pybullet_data

def generate_image(path, t, r):
    # connect to the physics simulation
    physicsClient = p.connect(p.DIRECT)

    # place the ground plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF('plane.urdf')

    # place a clamp
    clamp_id = p.loadSDF(path, globalScaling=1.0)[0]
    p.resetBasePositionAndOrientation(clamp_id, t, r)

    # set up the camera
    width = 1280
    height = 720

    camera_eye_pos = [0, 0, 0.35]
    camera_target_pos = [0, 0, 0]
    camera_up = [1, 0, 0]
    view_matrix = p.computeViewMatrix(camera_eye_pos, camera_target_pos, camera_up)

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
    return rgba

def main(args):
    # create the output directory
    if not os.path.exists("./output"):
        os.mkdir("./output")

    # create the json object that stores the marker positions
    data = {}

    img_name = "0001.png"
    data[img_name] = []

    translation = [0, 0, 0]
    rotation = [0, 0, 0, 1]

    # load the csv file containing the marker points and the paths
    paths = np.loadtxt("../data/objects/marked_clamps/clamp_1/marker.csv", delimiter=',', skiprows=1, usecols=[0], dtype=str)
    marker = np.loadtxt("../data/objects/marked_clamps/clamp_1/marker.csv", delimiter=',', skiprows=2, usecols=[2, 3, 4])

    # generate the image of the unmarked clamp
    unmarked = generate_image(paths[0], translation, rotation)[:, :, 0:3]
    unmarked = cv.cvtColor(unmarked, cv.COLOR_RGB2BGR)
    cv.imwrite("./output/" + img_name, unmarked)

    for i, path in enumerate(paths[1:]):
        # generate the image of the current marker
        marker = generate_image(path, translation, rotation)[:, :, 0:3]

        # generate the mask
        marker = cv.cvtColor(marker, cv.COLOR_RGB2HSV)
        marker[:, :, 0] = (marker[:, :, 0] + 5) % 180
        lower = np.array([0, 200, 200])
        upper = np.array([10, 255, 255])
        mask = cv.inRange(marker, lower, upper)

        # calculate the marker position
        num_labels, _, _, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 2:
            print("\nError: multiple marker detected\n")
            exit()
        elif num_labels == 1:
            # only the background label was found
            continue

        data[img_name].append({
            "id": i,
            "x": str(centroids[1].round().astype(int)[0]),
            "y": str(centroids[1].round().astype(int)[1])
        })

        # draw the marker onto the image
        cv.circle(unmarked, tuple(centroids[1].round().astype(int)), 2, [0, 255, 0], -1)

    with open("./output/data.json", "w") as outfile:
        json.dump(data, outfile, indent=2)

    cv.imshow("img", unmarked)
    cv.waitKey(0)


if __name__ == '__main__':
    main(sys.argv)
