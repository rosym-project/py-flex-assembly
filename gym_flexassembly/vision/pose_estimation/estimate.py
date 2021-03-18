import argparse
import tempfile
import time

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.rc('font', weight='bold', size=14)
import matplotlib.pyplot as plt

from gym_flexassembly.vision.pose_detection.bounding_box_regression.pose_direct import regress_depth_plane, visualize_plane
import gym_flexassembly.vision.pose_estimation.features as fts


def vec2str(vec):
    return str(list(map(lambda _v: f"{f'{_v:.4f}':>7}", vec)))

def orn2str(orn):
    as_euler = orn.as_euler('zyx', degrees=True)
    return str(list(map(lambda _v: f"{f'{_v:.2f}':>7}", as_euler)))

def display_depth_image(img_depth):
    img_depth = cv.convertScaleAbs(img_depth, alpha=0.3)
    img_depth = cv.equalizeHist(img_depth)
    img_depth = cv.applyColorMap(img_depth, cv.COLORMAP_JET)
    return img_depth


class FilterList:

    def __init__(self):
        decimation_filter = rs.decimation_filter(magnitude=2.0)
        threshold_filter = rs.threshold_filter()
        depth_to_disparity = rs.disparity_transform(transform_to_disparity=True)
        spatial_filter = rs.spatial_filter(smooth_alpha=0.5, smooth_delta=20.0, magnitude=2.0, hole_fill=0.0)
        temporal_filter = rs.temporal_filter(smooth_alpha=0.4, smooth_delta=20.0, persistence_control=1)
        hole_filling_filter = rs.hole_filling_filter(mode=1)
        disparity_to_depth = rs.disparity_transform(transform_to_disparity=False)

        self.filters = [decimation_filter,
                        threshold_filter,
                        depth_to_disparity,
                        spatial_filter,
                        temporal_filter,
                        hole_filling_filter,
                        disparity_to_depth]

    def process(self, frame_depth):
        for _filter in self.filters:
            frame_depth = _filter.process(frame_depth)
        return frame_depth

filter_aligned = FilterList()
filter_unaligned = FilterList()


def reorder(bb1, bb2):
    idx = []
    for pt in bb1:
        dists = list(map(lambda _pt: np.linalg.norm(pt - _pt), bb2))
        _idx = np.argmin(dists)

        if _idx in idx:
            raise RuntimeError(f'Ambiguous bb reordering {[*idx, _idx]}')

        idx.append(np.argmin(dists))
    return bb2[idx]


def compute_planes(img_depth, bb, visualize=False):
    mask = np.zeros(img_depth.shape, np.uint8)
    cv.drawContours(mask, [bb.as_int()], -1, 255, -1)
    mask = mask == 255

    mask_outer = np.zeros(img_depth.shape, np.uint8)
    cv.drawContours(mask_outer, [bb.copy().scale(1.5, 2.0, move_center=False).as_int()], 0, 255, -1)
    cv.drawContours(mask_outer, [bb.as_int()], 0, 0, -1)
    mask_outer = mask_outer == 255

    interval_upper = [0.35, 0.65]
    interval_lower = [0.4, 0.6]

    plane_upper = regress_depth_plane(mask, img_depth, interval_upper)
    plane_lower = regress_depth_plane(mask_outer, img_depth, interval_lower)

    if visualize:
        #TODO: comment in for faster runtime
        # return plane_upper, plane_lower, np.zeros((100, 100, 3), np.uint8)
        dpi = 100
        figure_size = (img_depth.shape[1] / dpi, img_depth.shape[0] / dpi)
        fig = plt.figure(figsize=figure_size)
        ax = fig.gca(projection='3d', position=[0.01, 0.01, 0.98, 0.98])
        visualize_plane(mask, img_depth, interval_upper, plane_upper, ax)
        visualize_plane(mask_outer, img_depth, interval_lower, plane_lower, ax)
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            fig.savefig(f.name)
            img_fig = cv.imread(f.name, cv.IMREAD_COLOR)
        plt.close(fig)

        return plane_upper, plane_lower, img_fig

    return plane_upper, plane_lower


def get_order(_pts, _side):
    if np.linalg.norm(_pts[0] - _pts[1]) < np.linalg.norm(_pts[1] - _pts[2]):
        if _side == 0:
            return [1, 2, 0]
        else:
            return [2, 1, 3]
    else:
        if _side == 0:
            return [2, 3, 1]
        else:
            return [3, 2, 0]


def vis_pose(pos, orn, intrin, img):
    tvec = pos
    rvec, _ = cv.Rodrigues(orn.as_matrix())
    cam_matrix = np.array([[intrin.fx,         0, intrin.ppx],
                           [        0, intrin.fy, intrin.ppy],
                           [        0,         0,          1]])
    points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]) / 25
    points, _ = cv.projectPoints(points, rvec, tvec, cam_matrix, distCoeffs=None)
    points = np.intp(points[:, 0, :])

    cv.circle(img, tuple(points[-1]), 2, (0, 0, 0), -1)
    cv.line(img, tuple(points[-1]), tuple(points[0]), (127, 127, 0), 3)
    cv.line(img, tuple(points[-1]), tuple(points[1]), (0, 255, 0), 2)
    cv.line(img, tuple(points[-1]), tuple(points[2]), (0, 0, 255), 2)
    return img


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default=None)
args = parser.parse_args()

cv.namedWindow('Display', cv.WINDOW_NORMAL)
cv.resizeWindow('Display', 1600, 900)

# create object to align depth and color images
align = rs.align(rs.stream.color)

config = rs.config()
if args.file:
    config.enable_device_from_file(args.file)

pipeline = rs.pipeline()
pipeline.start(config)

height = 0
w_height = 1.0

pos = np.array([0, 0, 0])
w_pos = 1.0

try:
    while True:
        _since = time.time()
        # get frames aligned and unaligned
        frames_unaligned = pipeline.wait_for_frames()
        frames_aligned = align.process(frames_unaligned)

        # first process aligned frames
        frame_color = frames_aligned.get_color_frame()
        frame_depth = frames_aligned.get_depth_frame()
        # post-process depth frame
        frame_depth = filter_aligned.process(frame_depth)

        # retrieve color and depth images
        img_color = np.asanyarray(frame_color.get_data())
        img_color = cv.cvtColor(img_color, cv.COLOR_RGB2BGR)
        img_depth = np.asanyarray(frame_depth.get_data())

        # detect bb in aligned image
        bb_aligned = fts.detect_bounding_box(img_depth)
        # scale bb to fit color image (post-processing modifies size)
        scale_h = img_color.shape[1] / img_depth.shape[1]
        scale_w = img_color.shape[0] / img_depth.shape[0]
        bb_aligned.scale(scale_h, scale_w)

        # refine bb for color image
        bb_refined = fts.refine_bb(bb_aligned, img_color)
        try:
            # detect side with hole
            side = fts.detect_side(img_color, bb_refined.as_points())
        except RuntimeError:
            print('Could not detect side')
            continue
        # visualize bb and side with hole
        img_color = fts.visualize_features(img_color, bb_refined, side)


        # process unaligned depth frame
        frame_depth = frames_unaligned.get_depth_frame()
        filter_unaligned.process(frame_depth)

        bb_viz = np.zeros(img_color.shape, np.uint8)
        # retrieve depth image
        img_depth = np.asanyarray(frame_depth.get_data())
        # detect bb in unaligned depth image
        bb_unaligned = fts.detect_bounding_box(img_depth, viz=bb_viz)

        # compute planes through depth values in bbs (upper = clamp, lower = table) 
        plane_upper, plane_lower, img_planes = compute_planes(img_depth, bb_unaligned, visualize=True)
        img_planes = cv.resize(img_planes, img_color.shape[:2][::-1], cv.INTER_LINEAR)

        # visualize bb in dept image
        img_depth = display_depth_image(img_depth)
        cv.drawContours(img_depth, [bb_unaligned.as_int()], -1, (0, 255, 0), 2)
        img_depth = cv.resize(img_depth, img_color.shape[:2][::-1])

        # estimate hight as dist of bb center at upper and lower plane
        bb_unaligned_c = np.mean(bb_aligned.as_points(), axis=0)
        new_height = plane_lower.predict([bb_unaligned_c])[0] - plane_upper.predict([bb_unaligned_c])[0]
        new_height = new_height / 1000
        if new_height < 0:
            # negative height is a clue that the bb detection did not find the clamp
            continue
        # average over 10 latest values
        height = (1.0 - w_height) * height + w_height * new_height 
        w_height = max(0.1, w_height - 0.1)
        print(f'Height estimate {height:.3f}m')

        # reorder pts in unaligned bb to match points in aligned bb
        pts_color = bb_refined.as_int()
        try:
            pts_ordered = reorder(pts_color, bb_unaligned.as_points())
        except RuntimeError as e:
            print('Reordering error for {e}')
            continue
        # determine order to compute x- & y-directions correctly
        _order = get_order(pts_color, side)

        # visualize x and y in color image
        dir_x = pts_color[_order[0]] - pts_color[_order[1]]
        dir_y = pts_color[_order[2]] - pts_color[_order[0]]
        c = np.mean(bb_refined.as_points(), axis=0).astype(np.int)
        cx = (c + 0.25 * dir_x).astype(np.int)
        cy = (c + 0.25 * dir_y).astype(np.int)
        cv.line(img_color, tuple(c), tuple(cx), (255, 0, 0), 2)
        cv.line(img_color, tuple(c), tuple(cy), (0, 255, 0), 2)

        # compute 3d coordinates for unaligned bb
        intrin = frame_depth.get_profile().as_video_stream_profile().get_intrinsics()
        pts3d = list(map(lambda pt: rs.rs2_deproject_pixel_to_point(intrin, pt, plane_upper.predict([pt])[0] / 1000), pts_ordered))
        pts3d = np.array(pts3d)

        # compute x-, y- and z-directions in 3d
        dir_x = pts3d[_order[0]] - pts3d[_order[1]]
        dir_y = pts3d[_order[2]] - pts3d[_order[0]]
        dir_z = np.cross(dir_x, dir_y)
        if dir_z[2] < 0:
            # This should not happen -> only if one of x and z points in the wrong direction
            print(f'Flip z dir from {vec2str(dir_z / np.linalg.norm(dir_z))}')
            dir_z = np.cross(dir_y, dir_x)
        dir_z = dir_z / np.linalg.norm(dir_z)

        # compute pose from bb and directions and average
        new_pos = np.mean(pts3d, axis=0) + 0.5 * height * dir_z
        pos = (1.0 - w_pos) * pos + w_pos * new_pos 
        w_pos = max(0.1, w_pos - 0.1)

        rot_matrix = np.array([dir_x / np.linalg.norm(dir_x),
                               dir_y / np.linalg.norm(dir_y),
                               dir_z])
        orn = R.from_matrix(rot_matrix)
        print(f'Pos {vec2str(pos)}')
        print(f'Orn {orn2str(orn)}')

        to = time.time() - _since
        print(f'Loop took {to * 1000}ms')
        print()

        # visualize pose in depth image
        vis_pose(pos, orn, intrin, img_depth)


        to_show_upper = np.hstack((img_depth, img_color))
        to_show_lower = np.hstack((img_planes, bb_viz))
        cv.imshow('Display', np.vstack((to_show_upper, to_show_lower)))
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            while (cv.waitKey(0) != ord('p')):
                continue
finally:
    pipeline.stop()
    cv.destroyAllWindows()
