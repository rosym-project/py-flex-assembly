import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from gym_flexassembly.vision.pose_detection.projection.estimator import as_transform, FilterList
import gym_flexassembly.vision.pose_detection.projection.visualize as vis
import gym_flexassembly.vision.pose_detection.projection.features as fts

def as_transform(pos, orn):
    """
    Convert pose into transform as used by pytransform3d
    """
    # get rotation as wxyz quaternion
    orn_q = orn.as_quat()
    orn_q = [orn_q[-1], *orn_q[:-1]]

    # create pose and return transform
    _pose = np.hstack((pos, orn_q))
    return pt.transform_from_pq(_pose)

# TODO
# * Gegeben Pose des Arms:
# * Füge einige Koordinaten des Tisches in der Nähe darunter in Welt-Koords ein
# * Teste mit der Realsense Funktion, ob die Punkte auf dem Tiefenbild sichtbar sind, und ob die Abstände übereinstimmen...

# create object to align depth and color images
config = rs.config()
#config.enable_device_from_file('./realsense_recordings/bags_2/new_01.bag')
config.enable_device_from_file('./realsense_recordings/multiple_clamps.bag')
pipeline = rs.pipeline()
pipeline.start(config)

tm = TransformManager()
# init transformation form arm to cam
# pos_arm_in_world = np.array([0.0, 0.0, 1.0])
# orn_arm_in_world = R.from_quat([0.0, 0.0, 0.0, 1.0])
# world2arm = as_transform(pos_arm_in_world, orn_arm_in_world)
# tm.add_transform("arm", "world", world2arm)
# z = (550.15 - 467.00) / 1000 
# z = z - 0.026
# calib_pt_arm = np.array([-333.72, -491.74, 373.78])
# calib_pt_cam = np.array([-407.27, -537.22, 539.82])
# pos_cam_in_arm = (calib_pt_arm - calib_pt_cam) / 1000
# pos_cam_in_arm = R.from_euler('zyx', [0, 180, 0], degrees=True).apply(pos_cam_in_arm)
# pos_cam_in_arm[2] = z
# orn_cam_in_arm = R.from_euler('zyx', [45 + 90, 0, 0], degrees=True)
# arm2cam = as_transform(pos_cam_in_arm, orn_cam_in_arm)
# tm.add_transform("cam", "arm", arm2cam)

# set real arm pose
# pos_arm = np.array([-301.44, -430.66, 618.01]) / 1000
# orn_arm = R.from_euler('zyx', [-137.15, 0, 179.99], degrees=True)
# world2arm = as_transform(pos_arm, orn_arm)
# tm.add_transform("arm", "world", world2arm)

table_height = 0.025

pos_cam = np.array([-0.3, -0.3, 0.544 + table_height])
orn_cam = R.from_euler('zyx', [90, 180, 0], degrees=True)
tm.add_transform("cam", "world", as_transform(pos_cam, orn_cam))
print(pos_cam)

pos_table = tm.get_transform('cam', 'world')[:3, -1].copy()
print('pos table', pos_table)
pos_table[2] = table_height + 0.01

_pos_table = pos_table + np.array([0.05, 0.05, 0])
print('_pos table', _pos_table)
orn_table = R.from_quat([0, 0, 0, 1])
tm.add_transform("table1", "world", as_transform(_pos_table, orn_table))

_pos_table = pos_table + np.array([-0.05, 0.05, 0])
tm.add_transform("table2", "world", as_transform(_pos_table, orn_table))

_pos_table = pos_table + np.array([0.05, -0.05, 0])
tm.add_transform("table3", "world", as_transform(_pos_table, orn_table))


# ax = tm.plot_frames_in('world', s=0.2)
# ax.set_xlim((-0.7, 0))
# ax.set_ylim((-0.7, 0))
# ax.set_zlim((0, 0.75))
# plt.show()
# exit(0)


# Nico <nrabethge@techfak.uni-bielefeld.de>
# Mara <mbrandt@techfak.uni-bielefeld.de>
# Franz <franz@techfak.uni-bielefeld.de>

fl = FilterList()

cv.namedWindow('Display', cv.WINDOW_NORMAL)
cv.resizeWindow('Display', 1600, 900)

while True:
    frames = pipeline.wait_for_frames()
    frame_color = frames.get_color_frame()
    frame_depth = frames.get_depth_frame()
    frame_depth = fl.process(frame_depth)

    intrin = frame_depth.get_profile().as_video_stream_profile().get_intrinsics()

    pos_tabl1_in_cam = tm.get_transform('table1', 'cam')[:3, -1]
    pos_tabl2_in_cam = tm.get_transform('table2', 'cam')[:3, -1]
    pos_tabl3_in_cam = tm.get_transform('table3', 'cam')[:3, -1]

    as_pixel1 = np.array(rs.rs2_project_point_to_pixel(intrin, pos_tabl1_in_cam))
    as_pixel2 = np.array(rs.rs2_project_point_to_pixel(intrin, pos_tabl2_in_cam))
    as_pixel3 = np.array(rs.rs2_project_point_to_pixel(intrin, pos_tabl3_in_cam))
    print('As Pixel', as_pixel1)

    xs = np.array([as_pixel1[0], as_pixel2[0], as_pixel3[0]])
    ys = np.array([as_pixel1[1], as_pixel2[1], as_pixel3[1]])
    zs = [pos_tabl1_in_cam[2], pos_tabl2_in_cam[2], pos_tabl3_in_cam[2]]
    print('z before', zs)
    zs = np.array(zs) * 1000
    print('z after', zs)
    inputs = np.array([xs, ys]).T
    from sklearn import linear_model
    model = linear_model.LinearRegression().fit(inputs, zs)

    # retrieve color and depth images
    img_color = np.asanyarray(frame_color.get_data())
    img_color = cv.cvtColor(img_color, cv.COLOR_RGB2BGR)
    img_depth = np.asanyarray(frame_depth.get_data())

    xx = np.arange(img_depth.shape[1])[::-1]
    yy = np.arange(img_depth.shape[0])[::-1]
    xx, yy = np.meshgrid(xx, yy)
    plane = model.intercept_ + model.coef_[1] * xx + model.coef_[0] * yy

    _x = np.arange(xs.min() - 100, xs.max() + 100, (xs.max() + 200 - xs.min()) / 11)
    _y = np.arange(ys.min() - 100, ys.max() + 100, (200 + ys.max() - ys.min()) / 11)
    xx, yy = np.meshgrid(_x, _y)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_zlim([510, 530])
    ax.scatter(xs, ys, zs, s=50)
    inputs = np.array([xx.ravel(),yy.ravel()])
    outputs = model.predict(inputs.T).reshape(xx.shape)
    ax.plot_surface(xx, yy, outputs, rstride=1, cstride=1, alpha=0.2)
    # plt.show()
    
    # Only consider parts of the depth image over the table plane
    img_depth = np.where(plane < img_depth, 0, img_depth)
    img = img_depth / img_depth.max()
    img = np.where(img > 0.1, 255, 0).astype(np.uint8)
    # perform closing to fill small holes in depth img
    img = cv.dilate(img, np.ones((3, 3)), iterations=2)
    img = cv.erode(img, np.ones((3, 3)), iterations=2)
    # find boxes
    cnts, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bbs = list(map(lambda cnt: fts.BoundingBox(cv.minAreaRect(cnt)), cnts))

    def ratio_filter(bb):
        long_side_length = max(bb.w, bb.h)
        short_side_length = min(bb.w, bb.h)
        ratio = long_side_length / short_side_length
        return 3 < ratio < 6

    bbs = list(filter(ratio_filter, bbs))

    scale_h = img_color.shape[1] / img_depth.shape[1]
    scale_w = img_color.shape[0] / img_depth.shape[0]

    img_depth = vis.display_depth_image(img_depth)
    img_depth = cv.resize(img_depth, img_color.shape[:2][::-1])

    print(scale_h, scale_w)

    for bb in bbs:
        bb.scale(scale_h, scale_w)
        print(bb.as_int())
        cv.imshow('test', img_depth[bb.as_slice()])
        cv.drawContours(img_depth, [bb.as_int()], -1, (0, 255, 0), 3)

        cv.imshow('Display', np.vstack((img_color, img_depth)))

        print(bb)
        bb_refined = fts.refine_bb(bb, img_color)
        cv.drawContours(img_color, [bb_refined.as_int()], -1, (0, 255, 0), 3)

        cv.imshow('Display', np.vstack((img_color, img_depth)))

        if cv.waitKey(0) == ord('q'):
            break

