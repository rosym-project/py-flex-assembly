import argparse
import io
import tempfile
import time

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.rc('font', weight='bold', size=14)
import matplotlib.pyplot as plt

from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from gym_flexassembly.vision.pose_detection.bounding_box_regression.pose_direct import regress_depth_plane, visualize_plane
import gym_flexassembly.vision.pose_estimation.features as fts

# OFFSET_ARM_TO_GRIPPER = np.array([0, 0, 0.215])
# #pos_arm = np.array([-0.32364, -0.41404, 0.61695])
# #orn_arm = R.from_euler('zyx', [131.04, 0.0, 180], degrees=True)

# #pos_arm = np.array([-0.32364, -0.41404, 0.61695])
# orn_arm = R.from_euler('zyx', [131.04, 0.01, -180], degrees=True)

def fig2img(fig, dpi=100):
    with io.BytesIO() as io_buf:
        fig.savefig(io_buf, format='raw', dpi=dpi)
        io_buf.seek(0)
        img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        return cv.cvtColor(img[:, :, :3], cv.COLOR_RGB2BGR)

def as_wxyz_quat(orn):
    quat = orn.as_quat()
    return [quat[-1], *quat[:-1]]

# record.bag
# pos_arm = [-0.29899, -0.34657, 0.71081]
# orn_arm = R.from_euler('zyx', [131.04, 0, -179.99], degrees=True)

# bags_2/new_01.bag
pos_arm = np.array([-301.44, -430.66, 618.01]) / 1000
orn_arm = R.from_euler('zyx', [-137.15, 0, 179.99], degrees=True)
# bags_2/new_02.bag
# pos_arm = np.array([-273.24, -553.63, 618.50]) / 1000
# orn_arm = R.from_euler('zyx', [-113.92, 0.01, 180], degrees=True)

# position of placement 
pos_pla = np.array([-301.44, -430.66, 265.72 - 215]) / 1000
orn_pla = R.from_euler('zyx', [-137.15, 0, 179.99], degrees=True)

orn_arm = as_wxyz_quat(orn_arm)
world2arm = pt.transform_from_pq(np.hstack((pos_arm, orn_arm)))
# world2pla = pt.transform_from_pq(np.hstack((pos_pla, as_wxyz_quat(orn_pla))))

arm_offset_to_gripper = [0, 0, 0.215]
orn_gripper = R.from_quat([0, 0, 0, 1])
orn_gripper = as_wxyz_quat(orn_gripper)
arm2gripper = pt.transform_from_pq(np.hstack((arm_offset_to_gripper, orn_gripper)))

camera_offset_to_gripper = [-0.085, 0, 0.056]
orn_gripper_to_arm = R.from_euler('zyx', [45, 0, 0], degrees=True)
cam_offset_to_arm = orn_gripper_to_arm.apply(camera_offset_to_gripper)

orn_cam_to_arm = R.from_euler('zyx', [135, 0, 0], degrees=True)
orn_cam_to_arm = as_wxyz_quat(orn_cam_to_arm)
arm2cam = pt.transform_from_pq(np.hstack((cam_offset_to_arm, orn_cam_to_arm)))

pos_clamp_in_arm = np.array([-0.0882, -0.0584, 0.5842])
orn_clamp_in_arm = R.from_euler('zyx', [153.91, 5.53, 13.90], degrees=True)
orn_clamp_in_arm = as_wxyz_quat(orn_clamp_in_arm)
cam2clamp = pt.transform_from_pq(np.hstack((pos_clamp_in_arm, orn_clamp_in_arm)))

tm = TransformManager()
tm.add_transform("arm", "world", world2arm)
# tm.add_transform("placement", "world", world2pla)
tm.add_transform("cam", "arm", arm2cam)
# tm.add_transform("gripper", "arm", arm2gripper)


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
        threshold_filter = rs.threshold_filter(max_dist=0.8)
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
                        #hole_filling_filter,
                        disparity_to_depth]

    def process(self, frame_depth):
        for _filter in self.filters:
            frame_depth = _filter.process(frame_depth)
        return frame_depth

filter_aligned = FilterList()
filter_unaligned = FilterList()


def reorder(bb1, bb2):
    center_bb1 = np.array(bb1.as_rotated_rect()[0])
    center_bb2 = np.array(bb2.as_rotated_rect()[0])
    diff = center_bb1 - center_bb2

    pts_bb1 = bb1.as_points()
    pts_bb2 = bb2.as_points() + diff # move bb2 on the same center

    idx = []
    for pt in pts_bb1:
        dists = list(map(lambda _pt: np.linalg.norm(pt - _pt), pts_bb2))
        _idx = np.argmin(dists)

        if _idx in idx:
            raise RuntimeError(f'Ambiguous bb reordering {[*idx, _idx]}')

        idx.append(np.argmin(dists))
    return pts_bb2[idx] - center_bb1 + center_bb2


def compute_planes(img_depth, bb, ax=None):
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

    if ax is not None:
    # if visualize:
        # #TODO: comment in for faster runtime
        # # return plane_upper, plane_lower, np.zeros((100, 100, 3), np.uint8)
        # dpi = 100
        # figure_size = (img_depth.shape[1] / dpi, img_depth.shape[0] / dpi)
        # fig = plt.figure(figsize=figure_size)
        # ax = fig.gca(projection='3d', position=[0.01, 0.01, 0.98, 0.98])
        visualize_plane(mask, img_depth, interval_upper, plane_upper, ax)
        visualize_plane(mask_outer, img_depth, interval_lower, plane_lower, ax)

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
    cv.line(img, tuple(points[-1]), tuple(points[0]), (0, 0, 255), 3)
    cv.line(img, tuple(points[-1]), tuple(points[1]), (0, 255, 0), 2)
    cv.line(img, tuple(points[-1]), tuple(points[2]), (255, 0, 0), 2)
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

rot_matrix = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
w_rot = 1.0



try:
    fig = plt.figure()
    ax_planes = fig.add_subplot(1, 2, 1, projection='3d')
    ax_transforms = fig.add_subplot(1, 2, 2, projection='3d')

    imgs = {'color': None,
            'depth': None,
            'bbdet': None}

    def reset_axes():
        ax_planes.clear()

        ax_transforms.clear()
        ax_transforms.set_xlim((-0.75, 0.05))
        ax_transforms.set_ylim((-0.75, 0.05))
        ax_transforms.set_zlim((0.0, 1.0))

        plt.tight_layout(pad=2, w_pad=2, h_pad=0)

    while True:
        reset_axes()

        try:
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

            imgs['color'] = img_color
            imgs['bbdet'] = np.zeros((*img_depth.shape, 3), np.uint8)
            imgs['depth'] = display_depth_image(img_depth)

            bb_aligned = fts.detect_bounding_box(fts.flatten(img_depth), viz=imgs['bbdet'])

            # scale bb to fit color image (post-processing modifies size)
            scale_h = img_color.shape[1] / img_depth.shape[1]
            scale_w = img_color.shape[0] / img_depth.shape[0]
            bb_aligned.scale(scale_h, scale_w)

            # refine bb for color image
            bb_refined = fts.refine_bb(bb_aligned, img_color)
            side = fts.detect_side(img_color, bb_refined.as_points())
            cv.drawContours(imgs['color'], [bb_refined.as_int()], -1, (255, 0, 0), 3)
            fts.visualize_features(imgs['color'], bb_refined, side)

            # process unaligned depth frame
            frame_depth = frames_unaligned.get_depth_frame()
            frame_depth = filter_unaligned.process(frame_depth)

            # retrieve depth image
            img_depth = np.asanyarray(frame_depth.get_data())

            imgs['depth'] = display_depth_image(img_depth)
            bb_unaligned = fts.detect_bounding_box(fts.flatten(img_depth), viz=imgs['bbdet'])
            # visualize bb in depth image
            cv.drawContours(imgs['depth'], [bb_unaligned.as_int()], -1, (0, 255, 0), 2)

            # compute planes through depth values in bbs (upper = clamp, lower = table) 
            plane_upper, plane_lower = compute_planes(img_depth, bb_unaligned, ax=ax_planes)

            # estimate hight as dist of bb center at upper and lower plane
            bb_unaligned_c = np.array(bb_unaligned.as_rotated_rect()[0])[::-1]
            new_height = plane_lower.predict([bb_unaligned_c])[0] - plane_upper.predict([bb_unaligned_c])[0]
            new_height = new_height / 1000
            if new_height < 0:
                # negative height is a clue that the bb detection did not find the clamp
                raise RuntimeError(f'Predicted negative height {new_height}')
            # average over 10 latest values
            height = (1.0 - w_height) * height + w_height * new_height 
            w_height = max(0.1, w_height - 0.1)
            print(f'Height estimate {height:.3f}m from {new_height:.3f}m')

            # reorder pts in unaligned bb to match points in aligned bb
            pts_color = bb_refined.as_int()
            pts_ordered = reorder(bb_refined, bb_unaligned)
            _order = get_order(pts_color, side)

            # visualize x and y in color image
            dir_x = pts_color[_order[0]] - pts_color[_order[1]]
            dir_y = pts_color[_order[2]] - pts_color[_order[0]]
            c = np.mean(bb_refined.as_points(), axis=0).astype(np.int)
            cx = (c + 0.25 * dir_x).astype(np.int)
            cy = (c + 0.25 * dir_y).astype(np.int)
            cv.line(imgs['color'], tuple(c), tuple(cx), (255, 0, 0), 2)
            cv.line(imgs['color'], tuple(c), tuple(cy), (0, 255, 0), 2)
            # visualize sides used for dir computation
            cv.line(imgs['depth'], tuple(pts_ordered[_order[0]].astype(int)), tuple(pts_ordered[_order[1]].astype(int)), (255, 0, 0), 3)
            cv.line(imgs['depth'], tuple(pts_ordered[_order[0]].astype(int)), tuple(pts_ordered[_order[2]].astype(int)), (0, 0, 255), 3)

            # compute 3d coordinates for unaligned bb
            intrin = frame_depth.get_profile().as_video_stream_profile().get_intrinsics()
            pts3d = list(map(lambda pt: rs.rs2_deproject_pixel_to_point(intrin, pt, plane_upper.predict([pt])[0] / 1000), pts_ordered))
            pts3d = np.array(pts3d)

            # compute x-, y- and z-directions in 3d
            dir_x = pts3d[_order[0]] - pts3d[_order[1]]
            dir_y = pts3d[_order[2]] - pts3d[_order[0]]
            dir_z = np.cross(dir_x, dir_y)
            dir_z = dir_z / np.linalg.norm(dir_z)
            if dir_z[2] < 0:
                print('Recompute z dir...')
                # z directory should point away from the camera which means z needs to be negative
                dir_z = -dir_z

                # recompute y directions because of change z direction
                ly = np.linalg.norm(dir_y)
                dir_y = np.cross(dir_z, dir_x)
                dir_y = ly * dir_y / np.linalg.norm(dir_y)

            # compute pose from bb and directions and average
            new_pos = np.mean(pts3d, axis=0) + 0.5 * height * dir_z
            pos = (1.0 - w_pos) * pos + w_pos * new_pos 
            w_pos = max(0.1, w_pos - 0.1)

            # compute rotations from directions
            new_rot_matrix = np.array([dir_x / np.linalg.norm(dir_x),
                                       dir_y / np.linalg.norm(dir_y),
                                       dir_z]).T
            rot_matrix = (1.0 - w_rot) * rot_matrix + w_rot * new_rot_matrix
            _x = rot_matrix[:, 0]
            _y = rot_matrix[:, 1]
            _z = rot_matrix[:, 2]

            _z = np.cross(_x, _y)
            _y = np.cross(_z, _x)
            rot_matrix = np.array([_x / np.linalg.norm(_x),
                                   _y / np.linalg.norm(_y),
                                   _z / np.linalg.norm(_z)]).T


            print(f'X {dir_x / np.linalg.norm(dir_x)}')
            print(f'Y {dir_y / np.linalg.norm(dir_y)}')
            print(f'Z {dir_z / np.linalg.norm(dir_z)}')
            print(f'{rot_matrix}')
            orn = R.from_matrix(rot_matrix)
            print(f'Pos {vec2str(pos)}')
            print(f'Orn {orn2str(orn)}')

            # enter new pose in transfrom manager in visualize
            cam2clamp = pt.transform_from_pq(np.hstack((pos, as_wxyz_quat(orn))))
            tm.add_transform("clamp", "cam", cam2clamp)
            tm.plot_frames_in('world', ax=ax_transforms, s=0.2)

            clamp_in_world = tm.get_transform('clamp', 'world')
            _pos = clamp_in_world[:3, -1] * 1000
            _orn = R.from_matrix(clamp_in_world[:3, :3])
            print(f'Pos World {vec2str(_pos)}')
            print(f'Orn World {orn2str(_orn)}')

            # visualize pose in depth image
            vis_pose(pos, orn, intrin, imgs['depth'])

            to = time.time() - _since
            print(f'Loop took {to * 1000}ms')
            print()
        except Exception as e:
            print(f'Something went wrong! - {e}')
            print()


        _img_color = imgs['color']
        _img_depth = cv.resize(imgs['depth'], _img_color.shape[:2][::-1])
        _img_bbdet = cv.resize(imgs['bbdet'], _img_color.shape[:2][::-1])
        _img_plots = cv.resize(fig2img(fig), _img_color.shape[:2][::-1])

        to_show_upper = np.hstack((_img_depth, _img_color))
        to_show_lower = np.hstack((_img_plots, _img_bbdet))
        cv.imshow('Display', np.vstack((to_show_upper, to_show_lower)))
        key = cv.waitKey(1000 // 25)
        # key = cv.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('p'):
            while (cv.waitKey(0) != ord('p')):
                continue
finally:
    pipeline.stop()
    cv.destroyAllWindows()
