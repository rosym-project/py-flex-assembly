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

# create object to align depth and color images
config = rs.config()
config.enable_device_from_file('./realsense_recordings/multiple_clamps.bag')
pipeline = rs.pipeline()
pipeline.start(config)

tm = TransformManager()
pos_cam = np.array([-0.3, -0.3, 0.544 + 0.025])
orn_cam = R.from_euler('zyx', [90, 180, 0], degrees=True)
tm.add_transform("cam", "world", as_transform(pos_cam, orn_cam))

fl = FilterList()

cv.namedWindow('Display', cv.WINDOW_NORMAL)
cv.resizeWindow('Display', 1600, 900)

depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
while True:
    frames = pipeline.wait_for_frames()
    frame_color = frames.get_color_frame()
    frame_depth = frames.get_depth_frame()
    frame_depth = fl.process(frame_depth).as_depth_frame()

    # bbs = list(filter(ratio_filter, bbs))
    bbs = fts.detect_bbs(tm, frame_depth)

    # retrieve color and depth images
    img_color = np.asanyarray(frame_color.get_data())
    img_color = cv.cvtColor(img_color, cv.COLOR_RGB2BGR)
    img_depth = np.asanyarray(frame_depth.get_data())

    scale_h = img_color.shape[1] / img_depth.shape[1]
    scale_w = img_color.shape[0] / img_depth.shape[0]

    img_depth = vis.display_depth_image(img_depth)

    for bb in bbs:
        bb_color = []
        cv.drawContours(img_depth, [bb.as_int()], -1, (0, 255, 0), 2)
        for _pt in bb.as_points():
            try:
                res = fts.depth_to_color_pixel(_pt, frame_depth, frame_color)
                _res = np.array(res).astype(int)
                bb_color.append(_res)
                cv.circle(img_color, tuple(_res), 5, (0, 255, 0), -1)

                res = fts.color_to_depth_pixel(res, frame_depth, frame_color, depth_scale)
                _res = np.array(res).astype(int)
                cv.circle(img_depth, tuple(_res), 3, (0, 0, 255), -1)
            except RuntimeError as e:
                print(e)
                print()
        bb_color = fts.BoundingBox(cv.minAreaRect(np.array(bb_color)))
        from gym_flexassembly.vision.pose_detection.projection.side_model import SidePredictor
        side = fts.detect_side_2(img_color, bb_color, SidePredictor('./models/side_model.pth'))

    img_depth = cv.resize(img_depth, img_color.shape[:2][::-1])
    cv.imshow('Display', np.vstack((img_color, img_depth)))


    # print(bb)
    # bb_refined = fts.refine_bb(bb, img_color)
    # cv.drawContours(img_color, [bb_refined.as_int()], -1, (0, 255, 0), 3)
    key = cv.waitKey(0)
    if key == ord('q'):
        exit(0)

