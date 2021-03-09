import time

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.spatial.transform import Rotation as R

from gym_flexassembly.vision.pose_detection.bounding_box_regression.extract_features_reality import detect_features, visualize_features

props = {
    'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],
    'fx': 929.2593994140625,
    'fy': 927.8283081054688,
    'height':  720,
    'width': 1280,
    'ppx': 648.0797119140625,
    'ppy': 370.44757080078125,
    'model': rs.distortion.inverse_brown_conrady}

camera_intrinsics = rs.intrinsics()
for key in props:
    setattr(camera_intrinsics, key, props[key])


def project_point(pt, depth):
    pt = rs.rs2_deproject_pixel_to_point(camera_intrinsics, pt, depth)
    return np.array(pt)


def to_2d(pt, tvec=None, rvec=None):
    if tvec is None:
        tvec = np.array(pt)

    if rvec is None:
        rvec, _ = cv.Rodrigues(R.from_quat([0, 0, 0, 1]).as_matrix())

    cam_matrix = np.array([[props['fx'],           0, props['ppx']],
                           [          0, props['fy'], props['ppy']],
                           [          0,           0,           1]])
    _pt, _ = cv.projectPoints(np.array([pt]), rvec, tvec, cam_matrix, distCoeffs=np.array(props['coeffs']))
    return _pt[0, 0, :]


def regress_depth_plane(mask, img_depth, interval=(0.25, 0.75)):
    if mask.dtype != np.bool:
        raise ValueError(f'Boolean mask and not {mask.dtype} required!')

    depths = img_depth[mask]
    xs, ys = np.nonzero(mask)

    args = np.argsort(depths)
    size = args.shape[0]
    args = args[int(interval[0] * size):int(interval[1] * size)]

    depths = depths[args]
    inputs = np.array([xs[args], ys[args]]).T
    
    return linear_model.LinearRegression().fit(inputs, depths)


def scale_box(box, scale):
    dir1 = box[0] - box[1]
    dir2 = box[2] - box[1]

    s = scale 
    return np.array([box[0] + s * dir1 - s * dir2,
                     box[1] - s * dir1 - s * dir2,
                     box[2] - s * dir1 + s * dir2,
                     box[3] + s * dir1 + s * dir2])


def visualize_plane(mask, img_depth, interval, depth_plane, ax, c='r'):
    depths = img_depth[mask]
    args = np.argsort(depths)
    args = args[int(interval[0] * args.shape[0]):int(interval[1] * args.shape[0])]
    xs, ys = np.nonzero(mask)

    _x = np.arange(xs.min(), xs.max(), (xs.max() - xs.min()) / 11)
    _y = np.arange(ys.min(), ys.max(), (ys.max() - ys.min()) / 11)
    xx, yy = np.meshgrid(_x, _y)

    ax.scatter(xs[args], ys[args], depths[args], c=c, s=50)
    inputs = np.array([xx.ravel(),yy.ravel()])
    outputs = depth_plane.predict(inputs.T).reshape(xx.shape)
    ax.plot_surface(xx, yy, outputs, rstride=1, cstride=1, alpha=0.2)


def compute_upper_plane(box, img_depth):
    mask = np.zeros(img_depth.shape, np.uint8)
    cv.drawContours(mask, [np.intp(box)], -1, 255, -1)
    mask = mask == 255
    interval = (0.3, 0.45)
    return regress_depth_plane(mask, img_depth, interval=interval)


def compute_upper_plane(box, img_depth):
    mask = np.zeros(img_depth.shape, np.uint8)
    cv.drawContours(mask, [np.intp(box)], -1, 255, -1)
    mask = mask == 255
    interval = (0.3, 0.45)
    return regress_depth_plane(mask, img_depth, interval=interval)


def compute_lower_plane(box, img_depth):
    mask_outer = np.zeros(img_depth.shape, np.uint8)
    cv.drawContours(mask_outer, [np.intp(scale_box(box, 0.2))], -1, 255, -1)
    cv.drawContours(mask_outer, [np.intp(box)], -1, 0, -1)
    mask_outer = mask_outer == 255
    return regress_depth_plane(mask_outer, img_depth)


def compute_clamp_height(box, depth_plane_upper, depth_plane_lower):
    pt = np.mean(box, axis=0)
    pt_upper = project_point(pt, depth_plane_upper.predict([pt])[0])
    pt_lower = project_point(pt, depth_plane_lower.predict([pt])[0])
    return np.linalg.norm(pt_upper - pt_lower)


def compute_bounding_box(box2d, depth_plane, height, feature_side):
    upper_rect = list(map(lambda pt: project_point(pt, depth_plane.predict([pt])[0]), box2d))

    # determine x and y direction based on longer side and side with hole
    dir1 = upper_rect[0] - upper_rect[1]
    dir2 = upper_rect[2] - upper_rect[1]
    if np.linalg.norm(dir1) > np.linalg.norm(dir2):
        if feature_side == 1:
            dir_x = dir1
            dir_y = dir2
        else:
            dir_x = -dir1
            dir_y = -dir2
    else:
        if feature_side == 1:
            dir_x = dir2
            dir_y = dir1
        else:
            dir_x = -dir2
            dir_y = -dir1

    # compute z direction from cross product
    dir_z = np.cross(dir_x, dir_y)
    dir_z = dir_z / np.linalg.norm(dir_z)

    # compute bottom coordinates and create 3d bounding box
    lower_rect = list(map(lambda pt: pt + dir_z * height, upper_rect))
    upper_rect.extend(lower_rect)
    return np.array(upper_rect), [dir_x, dir_y, dir_z]


def visualize_pose(pos, orn, img):
    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    axis *= 30
    rvec, _ = cv.Rodrigues(orn.inv().as_matrix())

    pts = list(map(lambda pt: to_2d(pt, tvec=pos, rvec=rvec), axis))
    pts = np.intp(pts)

    cv.circle(img, tuple(pts[-1]), 5, (255, 0, 0), -1)
    cv.line(img, tuple(pts[-1]), tuple(pts[0]), (255, 0, 0), 2)
    cv.line(img, tuple(pts[-1]), tuple(pts[1]), (0, 255, 0), 2)
    cv.line(img, tuple(pts[-1]), tuple(pts[2]), (0, 0, 255), 2)


def detect_pose(img_color, img_depth, visualize=True):
    # compute features (bounding rectangle and side with hole)
    box, features = detect_features(img_color, img_depth)
    # approximate depth values in rectangle as plane
    depth_plane = compute_upper_plane(box, img_depth)
    # approximate depth values outside rectangle as plane
    _depth_plane = compute_lower_plane(box, img_depth)

    # compute height of clamp as distance at bounding rectangle center of planes
    #TODO this computation seems to fail a lot
    # height = compute_clamp_height(box, depth_plane, _depth_plane)
    height = 50

    # compute 3d bounding box
    box3d, [dir_x, dir_y, dir_z] = compute_bounding_box(box, depth_plane, height, features[6])

    # compute pose as center of bounding box and orientation from directions
    pos = box3d.mean(axis=0)
    rot_matrix = np.array([dir_x / np.linalg.norm(dir_x),
                           dir_y / np.linalg.norm(dir_y),
                           dir_z / np.linalg.norm(dir_z)])
    orn = R.from_matrix(rot_matrix)
    
    # visualize pose
    if visualize:
        visualize_features(img_color, img_depth, box, features)
        visualize_pose(pos, orn, img_color)

        print(f'Detected height: {height:2f}mm')
        c = to_2d(box3d[:4].mean(axis=0))
        c = np.intp(c)
        cv.circle(img_color, tuple(c), 5, (255, 0, 0), -1)

    return pos, orn


if __name__ == '__main__':
    clamp_dir = 'pose_6'
    for i in range(75):
        # load image
        name = f'{i:05d}'
        img_depth = np.load(f'./datasets/real/initial/{clamp_dir}/{name}.npy')
        img_color = cv.imread(f'./datasets/real/initial/{clamp_dir}/{name}.png')

        pos, orn = detect_pose(img_color, img_depth)
        cv.imshow('Img', img_color)
        if cv.waitKey(0) == ord('q'):
            break

        # visualize planes
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # visualize_plane(mask, img_depth, interval, depth_plane, ax)
        # visualize_plane(mask_outer, img_depth, (0.25, 0.75), _depth_plane, ax)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.axis('auto')
        # ax.axis('tight')
        # plt.show()

