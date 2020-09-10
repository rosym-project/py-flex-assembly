import sys
import time

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
    width = 1080
    height = 720

    camera_eye_pos = [0, 0, 0.7]
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

# measures the distances between each pixel of an image and a reference color
# if a distance is smaller than max_dist the pixel is marked in the mask
def marker_mask(img, marker_color, max_dist):
    mask = np.zeros(img.shape[:2], dtype=np.byte)
    dists = img - np.full(img.shape, marker_color)

    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
            if np.linalg.norm(dists[r, c, :]) <= max_dist:
                mask[r, c] = 1

    return mask

def main(args):
    translation = [0, 0, 0]
    rotation = [0, 0, 0, 1]

    # generate the images
    unmarked = generate_image("../data/objects/marked_clamps/clamp_1/unmarked/clamp_1_unmarked.sdf", translation, rotation)[:, :, 0:3]
    unmarked = cv.cvtColor(unmarked, cv.COLOR_RGB2BGR)
    marked = generate_image("../data/objects/marked_clamps/clamp_1/marker_5/clamp_1_marker_5.sdf", translation, rotation)[:, :, 0:3]
    marked = cv.cvtColor(marked, cv.COLOR_RGB2BGR)

    # generate the mask
    marker_color = [0, 0, 255]
    max_dist = 40
    # mask = marker_mask(marked, marker_color, max_dist) * 255

    lower = np.array([0, 200, 200])
    upper = np.array([10, 255, 255])
    mask = cv.inRange(cv.cvtColor(marked, cv.COLOR_BGR2HSV), lower, upper)

    cv.imshow("mask", mask)
    cv.imshow("img", marked)
    cv.waitKey(0)

    # calculate the marker position
    num_labels, _, _, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 2:
        print("Error: multiple marker detected")
        return(-1)
    elif num_labels == 1:
        print("No marker found")
        #continue
    marker_pos = centroids[1]
    print(marker_pos/[1080, 720])

    # draw the marker onto the image
    cv.circle(unmarked, tuple(np.round(marker_pos).astype(np.int)), 2, [0, 255, 0], -1)
    cv.imshow("img", unmarked)
    cv.waitKey(0)


if __name__ == '__main__':
    main(sys.argv)
