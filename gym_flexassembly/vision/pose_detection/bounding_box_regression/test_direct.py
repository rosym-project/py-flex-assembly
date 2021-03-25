
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from scipy.spatial.transform import Rotation as R

from gym_flexassembly.vision.pose_detection.bounding_box_regression.pose_direct import detect_pose

ACCUM_WEIGHT = 0.1

pos_arm = np.array([-0.32364, -0.41404, 0.61695])
orn_arm = R.from_euler('zyx', [131.04, 0.0, 180], degrees=True)

camera_offset = [-0.085, 0, 0.056]
pos_cam = pos_arm + camera_offset
orn_cam = orn_arm

print(f'Arm pos {pos_arm}')
print(f'Arm orn {orn_arm.as_euler("zyx", degrees=True).astype(np.int)}')

print(f'Camera pos {pos_cam}')
print(f'Camera orn {orn_cam.as_euler("zyx", degrees=True).astype(np.int)}')

def get_wold_pose(pos, orn):
    _pos = orn_cam.inv().apply(pos) + pos_cam
    _orn = orn_cam.inv() * orn
    return _pos, _orn


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

def build_func(udepth, acolor):
    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
    depth_min = 0.11 #meter
    depth_max = 1.0 #meter

    depth_intrin = udepth.get_profile().as_video_stream_profile().get_intrinsics()
    color_intrin = acolor.get_profile().as_video_stream_profile().get_intrinsics()

    depth_to_color_extrin = udepth.get_profile().get_extrinsics_to(acolor.get_profile())
    color_to_depth_extrin = acolor.get_profile().get_extrinsics_to(udepth.get_profile())

    def func(color_point, depth_frame):
        return rs.rs2_project_color_pixel_to_depth_pixel(
                depth_frame.get_data(), depth_scale,
                depth_min, depth_max,
                depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, color_point)
    return func


def update_transform_func(transform_func, depth_frame):
    def func(color_point):
        return transform_func(color_point, depth_frame)
    return func


try:
    # fetch first image to create accumulated depth image
    frames_unaligned = pipeline.wait_for_frames()
    frames = align.process(frames_unaligned)

    transform_func = build_func(frames_unaligned.get_depth_frame(), frames.get_color_frame())

    frame_depth = np.asanyarray(frames.get_depth_frame().as_frame().get_data())
    accum_depth = np.float32(frame_depth)

    while True:
        frames_unaligned = pipeline.wait_for_frames()
        frames = align.process(frames_unaligned)

        tf = update_transform_func(transform_func, frames_unaligned.get_depth_frame())

        frame_depth = np.asanyarray(frames.get_depth_frame().as_frame().get_data())
        cv.accumulateWeighted(frame_depth, accum_depth, alpha=ACCUM_WEIGHT)

        frame_color = np.asanyarray(frames.get_color_frame().as_frame().get_data())
        frame_color = cv.cvtColor(frame_color, cv.COLOR_RGB2BGR)

        try:
            pos, orn = detect_pose(frame_color, accum_depth, tf)

            print(f'In camera pos {pos}')
            print(f'In camera orn {orn.as_euler("zyx", degrees=True).astype(np.int)}')

            pos, orn = get_wold_pose(pos, orn)
            print(f'In world pos {pos}')
            print(f'In world orn {orn.as_euler("zyx", degrees=True).astype(np.int)}')
        except Exception as e:
            # continue
            print('Could not detect clamp!', e)
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


