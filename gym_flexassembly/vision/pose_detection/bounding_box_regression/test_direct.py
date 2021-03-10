
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from scipy.spatial.transform import Rotation as R

from gym_flexassembly.vision.pose_detection.bounding_box_regression.pose_direct import detect_pose

ACCUM_WEIGHT = 0.25
# CAMERA_POS = np.array([-0.32364, -0.41404, 0.61695])
CAMERA_POS = [-0.19575, -0.37105, 0.61695]
CAMERA_ORN_EULER = [131.03, -0.01, 180.00]
# CAMERA_POS = [-0.31271, -0.41501, 0.61639]
# CAMERA_ORN_EULER = [131.23, 0, 180]
CAMERA_ORN = R.from_euler('zyx', CAMERA_ORN_EULER, degrees=True)

OFFSET_ARM_TO_GRIPPER = np.array([0.0, 0.0, 0.215])

def gripper_to_arm(pos, orn):
    _pos = pos - orn.apply(OFFSET_ARM_TO_GRIPPER)
    return _pos, orn

def get_wold_pose(pose_in_coords, pose_coords):
    pos, orn = pose_in_coords
    pos_c, orn_c = pose_coords

    pos = orn_c.inv().apply(pos) + pos_c
    orn = orn_c.inv() * orn
    return pos, orn


def estimate_pose(img, depth):
    # compute features
    box, feature_vec = detect_features(img, depth)

    # compute and print prediction
    pos_features = np.array([feature_vec])
    pos = estimator_pos.predict(pos_features[:, [0, 1, 2, 3, 4, 5, 6]])[0]
    print(f'Predicted position {format_arr(pos)}')

    orn_features = np.array(feature_vec)
    orn_features = np.hstack((orn_features, pos, CAMERA_ORN.as_quat()))
    # not all features are used by the current model
    used_features = [2, 4, 6, 8, 13, 14, 15, 16]
    # predict and normalize quaternion
    orn = estimator_orn.predict([orn_features[used_features]])[0]
    #orn_quat = orn / np.linalg.norm(orn)
    orn = R.from_quat(orn / np.linalg.norm(orn))

    print(f'Camera position:    {CAMERA_POS}')
    print(f'Camera orientation: {CAMERA_ORN.as_euler("zyx")}')
    print()
    print(f'Position Clamp in Arm:   {format_arr(pos)}')
    pos, orn = get_wold_pose((pos, orn), (CAMERA_POS, CAMERA_ORN))
    print(f'Position Clamp in World: {format_arr(pos)}')
    pos, orn = gripper_to_arm(pos, orn)
    print(f'Position Arm in World:   {format_arr(pos)}')
    orn_quat = orn.as_quat()
    orn_euler = orn.as_euler('zyx', degrees=True)
    
    print(f'Position:    {format_arr(pos)}')
    print(f'Orientation: {format_arr(orn_euler)} as quat {format_arr(orn_quat)}')
    print(f'Features:    {format_arr(feature_vec)}')
    print()

    # print and visualize features
    cv.imshow('Features', visualize(img, depth, box, feature_vec))


def display_depth_image(img_depth):
    img_depth = cv.convertScaleAbs(img_depth, alpha=0.3)
    img_depth = cv.equalizeHist(img_depth)
    img_depth = cv.applyColorMap(img_depth, cv.COLORMAP_JET)
    return img_depth

cv.namedWindow('Display', cv.WINDOW_NORMAL)
cv.resizeWindow('Display', 1920 // 2, 1080)

# create object to align depth and color images
align = rs.align(rs.stream.color)

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()
try:
    # fetch first image to create accumulated depth image
    frames = align.process(pipeline.wait_for_frames())

    frame_depth = np.asanyarray(frames.get_depth_frame().as_frame().get_data())
    accum_depth = np.float32(frame_depth)

    while True:
        frames = align.process(pipeline.wait_for_frames())

        frame_depth = np.asanyarray(frames.get_depth_frame().as_frame().get_data())
        cv.accumulateWeighted(frame_depth, accum_depth, alpha=ACCUM_WEIGHT)

        frame_color = np.asanyarray(frames.get_color_frame().as_frame().get_data())
        frame_color = cv.cvtColor(frame_color, cv.COLOR_RGB2BGR)

        try:
            pos, orn = detect_pose(frame_color, accum_depth)

            print(f'Pos {pos}')
            print(f'Orn {orn.as_euler("zyx", degrees=True).astype(np.int)}')
        except:
            # print('Could not detect clamp!')
            pass

        to_show = (frame_color, display_depth_image(accum_depth))
        to_show = np.concatenate(to_show, axis=0)

        # cv.imshow('Depth', display_depth_image(accum_depth))
        # cv.imshow('RGB', frame_color)
        cv.imshow('Display', to_show)

        key = cv.waitKey(1)
        if key == ord('q'):
            exit()
finally:
    pipeline.stop()
    cv.destroyAllWindows()


