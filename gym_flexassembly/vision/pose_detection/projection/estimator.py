import time

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from sklearn import linear_model

import matplotlib
import matplotlib.pyplot as plt

from pytransform3d import transformations as pt

import gym_flexassembly.vision.pose_detection.projection.features as fts
import gym_flexassembly.vision.pose_detection.projection.visualize as viz


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


def vec2str(vec):
    return str(list(map(lambda _v: f"{f'{_v:.4f}':>7}", vec)))


def orn2str(orn):
    as_euler = orn.as_euler('zyx', degrees=True)
    return str(list(map(lambda _v: f"{f'{_v:.2f}':>7}", as_euler)))


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


class Averaging():

    def __init__(self, window_size, initial_value=None):
       self.min_weight = 1.0 / window_size
       self.weight = 1.0
       self.value = initial_value

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        self.value = (1.0 - self.weight) * self.value + self.weight * new_value
        self.weight = max(self.weight - self.min_weight, self.min_weight)

        return self.value


class AveragingRotation(Averaging):

    def update(self, new_value):
        # re-compute new axes to be length 1
        _x = new_value[:, 0]
        _x = _x / np.linalg.norm(_x)
        _y = new_value[:, 1]
        _y = _y / np.linalg.norm(_y)
        _z = new_value[:, 2]
        _z = _z / np.linalg.norm(_z)
        new_value = np.array([_x, _y, _z]).T

        # compute new value as average
        return super().update(new_value)


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


def compute_planes(img_depth, bb, ax=None):
    mask = np.zeros(img_depth.shape, np.uint8)
    cv.drawContours(mask, [bb.as_int()], -1, 255, -1)
    mask = mask == 255

    mask_outer = np.zeros(img_depth.shape, np.uint8)
    cv.drawContours(mask_outer, [bb.copy().scale(1.5, 2.0, move_center=False).as_int()], 0, 255, -1)
    cv.drawContours(mask_outer, [bb.as_int()], 0, 0, -1)
    mask_outer = mask_outer == 255

    interval_upper = [0.2, 0.5]
    interval_lower = [0.4, 0.6]

    plane_upper = regress_depth_plane(mask, img_depth, interval_upper)
    plane_lower = regress_depth_plane(mask_outer, img_depth, interval_lower)

    if ax is not None:
        viz.visualize_plane(mask, img_depth, plane_upper, ax, interval=interval_upper)
        viz.visualize_plane(mask_outer, img_depth, plane_lower, ax, interval=interval_lower)

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


class PoseEstimator():

    def __init__(self, window_size=25, debug=True, transform_manager=None):
        self.filter_unaligned = FilterList()
        self.filter_aligned = FilterList()
        self.align = rs.align(rs.stream.color)

        self.height_averaging = Averaging(window_size)
        self.pos_averaging = Averaging(window_size)
        self.orn_averaging = AveragingRotation(window_size)

        self.debug = debug
        self.tm = transform_manager

        self.imgs = {'color': None, 'depth': None, 'bbdet': None, 'figure': None}
        if self.debug:
            self.fig = plt.figure()
            if self.tm is not None:
                self.ax_planes = self.fig.add_subplot(1, 2, 1, projection='3d')
                self.ax_transforms = self.fig.add_subplot(1, 2, 2, projection='3d')
            else:
                self.ax_planes = self.fig.add_subplot(1, 1, 1, projection='3d')


    def estimate(self, frames):
        if self.debug:
            self.reset_figure()
            since = time.time()

        frames_aligned = self.align.process(frames)
        bb_refined, side = self.process_aligned_frames(frames_aligned)

        pos, orn = self.process_unaligned_frames(frames, bb_refined, side)

        if self.debug:
            took = time.time() - since
            print(f'Estimation took {took * 1000:.3f}ms')

        return pos, orn

    def process_aligned_frames(self, frames):
        # first process aligned frames
        frame_color = frames.get_color_frame()
        frame_depth = frames.get_depth_frame()
        # post-process depth frame
        frame_depth = self.filter_aligned.process(frame_depth)

        # retrieve color and depth images
        img_color = np.asanyarray(frame_color.get_data())
        img_color = cv.cvtColor(img_color, cv.COLOR_RGB2BGR)
        img_depth = np.asanyarray(frame_depth.get_data())

        if self.debug:
            self.imgs['color'] = img_color
            self.imgs['depth'] = viz.display_depth_image(img_depth)
            self.imgs['bbdet'] = np.zeros((*img_depth.shape, 3), np.uint8)

            if self.imgs['figure'] is None:
                self.imgs['figure'] = np.zeros((*img_depth.shape, 3), np.uint8)

        bb_aligned = fts.detect_bounding_box(fts.flatten(img_depth), viz=self.imgs['bbdet'])

        # scale bb to fit color image (post-processing reduces size)
        scale_h = img_color.shape[1] / img_depth.shape[1]
        scale_w = img_color.shape[0] / img_depth.shape[0]
        bb_aligned.scale(scale_h, scale_w)

        # refine bb for color image
        bb_refined = fts.refine_bb(bb_aligned, img_color)
        if self.debug:
            cv.drawContours(self.imgs['color'], [bb_refined.as_int()], -1, (255, 0, 0), 3)

        side = fts.detect_side(img_color, bb_refined.as_points())
        if self.debug:
            fts.visualize_features(self.imgs['color'], bb_refined, side)

        return bb_refined, side

    def process_unaligned_frames(self, frames, bb_aligned, side):
        frame_depth = frames.get_depth_frame()
        frame_depth = self.filter_unaligned.process(frame_depth)

        # retrieve depth image
        img_depth = np.asanyarray(frame_depth.get_data())
        # find bb
        bb_unaligned = fts.detect_bounding_box(fts.flatten(img_depth), viz=self.imgs['bbdet'])

        if self.debug:
            # visualize bb in depth image
            self.imgs['depth'] = viz.display_depth_image(img_depth)
            cv.drawContours(self.imgs['depth'], [bb_unaligned.as_int()], -1, (0, 255, 0), 2)

        # compute planes through depth values in bbs (upper = clamp, lower = table) 
        if self.debug:
            plane_upper, plane_lower = compute_planes(img_depth, bb_unaligned, ax=self.ax_planes)
        else:
            plane_upper, plane_lower = compute_planes(img_depth, bb_unaligned)

        # estimate hight as dist of bb center at upper and lower plane
        bb_center = np.array(bb_unaligned.as_rotated_rect()[0])[::-1]
        new_height = plane_lower.predict([bb_center])[0]
        new_height = new_height - plane_upper.predict([bb_center])[0]
        new_height = new_height / 1000
        if new_height < 0:
            # negative height is a clue that the bb detection did not find the clamp
            raise RuntimeError(f'Predicted negative height {new_height}')
        # average over 10 latest values
        height = self.height_averaging.update(new_height)

        # reorder pts in unaligned bb to match points in aligned bb
        pts_color = bb_aligned.as_int()
        pts_ordered = reorder(bb_aligned, bb_unaligned)
        _order = get_order(pts_color, side)

        if self.debug:
            # visualize x and y in color image
            dir_x = pts_color[_order[0]] - pts_color[_order[1]]
            dir_y = pts_color[_order[2]] - pts_color[_order[0]]
            c = np.mean(bb_aligned.as_points(), axis=0).astype(np.int)
            cx = (c + 0.25 * dir_x).astype(np.int)
            cy = (c + 0.25 * dir_y).astype(np.int)
            cv.line(self.imgs['color'], tuple(c), tuple(cx), (255, 0, 0), 2)
            cv.line(self.imgs['color'], tuple(c), tuple(cy), (0, 255, 0), 2)
            # visualize sides used for dir computation
            cv.line(self.imgs['depth'], tuple(pts_ordered[_order[0]].astype(int)), tuple(pts_ordered[_order[1]].astype(int)), (255, 0, 0), 3)
            cv.line(self.imgs['depth'], tuple(pts_ordered[_order[0]].astype(int)), tuple(pts_ordered[_order[2]].astype(int)), (0, 0, 255), 3)

        # compute 3d coordinates for unaligned bb
        intrin = frame_depth.get_profile().as_video_stream_profile().get_intrinsics()
        pts3d_upper = list(map(lambda pt: rs.rs2_deproject_pixel_to_point(intrin, pt, plane_upper.predict([pt[::-1]])[0] / 1000), pts_ordered))
        pts3d_upper = np.array(pts3d_upper)

        pts3d_lower = list(map(lambda pt: rs.rs2_deproject_pixel_to_point(intrin, pt, plane_lower.predict([pt[::-1]])[0] / 1000), pts_ordered))
        pts3d_lower = np.array(pts3d_lower)

        # compute x-, y- and z-directions in 3d
        """
        User upper rect to determine the width and length of the clamp
        but use the lower plane for directions since it is more stable
        """ 
        dir_x_upper = pts3d_upper[_order[0]] - pts3d_upper[_order[1]]
        dir_x_lower = pts3d_lower[_order[0]] - pts3d_lower[_order[1]]
        dir_x = dir_x_lower / np.linalg.norm(dir_x_lower)
        dir_x = dir_x * np.linalg.norm(dir_x_upper)

        dir_y_upper = pts3d_upper[_order[2]] - pts3d_upper[_order[0]]
        dir_y_lower = pts3d_lower[_order[2]] - pts3d_lower[_order[0]]
        dir_y = dir_y_lower / np.linalg.norm(dir_y_lower)
        dir_y = dir_y * np.linalg.norm(dir_y_upper)

        dir_z = np.cross(dir_x, dir_y)
        dir_z = dir_z / np.linalg.norm(dir_z)
        if dir_z[2] < 0:
            # z directory should point away from the camera which means z needs to be negative
            dir_z = -dir_z

            # recompute y directions because of change z direction
            ly = np.linalg.norm(dir_y)
            dir_y = np.cross(dir_z, dir_x)
            dir_y = ly * dir_y / np.linalg.norm(dir_y)

        # compute pose from bb and directions and average
        new_pos = np.mean(pts3d_upper, axis=0) - 0.5 * height * dir_z
        pos = self.pos_averaging.update(new_pos)

        """
        NOTE: this is a magic number.
        We calibrated the color-camera in relation to the arm
        and not the depth camera. This is an offset in x-direction
        corresponding to this offset.
        """
        unmeasure_offset = np.array([-0.065, 0, 0])
        pos = pos + unmeasure_offset

        # compute rotations from directions
        new_rot_matrix = np.array([dir_x / np.linalg.norm(dir_x),
                                   dir_y / np.linalg.norm(dir_y),
                                   dir_z]).T
        rot_matrix = self.orn_averaging.update(new_rot_matrix)
        orn = R.from_matrix(rot_matrix)

        if self.debug:
            print(f'Height estimate {height:.3f}m from {new_height:.3f}m')
            print(f'Pos {vec2str(pos)}')
            print(f'Orn {orn2str(orn)}')

            # visualize pose in depth image
            viz.visualize_pose(pos - unmeasure_offset, orn, intrin, self.imgs['depth'])

            if self.tm is not None:
                self.tm.add_transform('clamp', 'cam', as_transform(pos, orn))
                self.tm.plot_frames_in('world', ax=self.ax_transforms, s=0.2)

            self.imgs['figure'] = viz.figure_to_img(self.fig)

        return pos, orn


    def reset_figure(self):
        self.ax_planes.clear()

        if self.tm is not None:
            self.ax_transforms.clear()
            self.ax_transforms.set_xlim((-0.75, 0.05))
            self.ax_transforms.set_ylim((-0.75, 0.05))
            self.ax_transforms.set_zlim((0.0, 1.0))

        plt.tight_layout(pad=2, w_pad=2, h_pad=0)


    def create_img(self):
        _img_color = self.imgs['color']
        _img_depth = cv.resize(self.imgs['depth'], _img_color.shape[:2][::-1])
        _img_bbdet = cv.resize(self.imgs['bbdet'], _img_color.shape[:2][::-1])
        _img_plots = cv.resize(self.imgs['figure'], _img_color.shape[:2][::-1])

        upper = np.hstack((_img_depth, _img_color))
        lower = np.hstack((_img_plots, _img_bbdet))
        return np.vstack((upper, lower))


if __name__ == '__main__':
    import argparse
    from pytransform3d.transform_manager import TransformManager

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default=None)
    parser.add_argument('--averaging_window', type=int, default=25,
                        help='the detected pose is averaged over this many frames')
    args = parser.parse_args()

    cv.namedWindow('Display', cv.WINDOW_NORMAL)
    cv.resizeWindow('Display', 1600, 900)

    # create object to align depth and color images
    config = rs.config()
    if args.file: config.enable_device_from_file(args.file)

    pipeline = rs.pipeline()
    pipeline.start(config)

    pos_b_in_world = np.array([0, 0, 0])
    orn_b_in_world = R.from_euler('zyx', [270, 0, 0], degrees=True)
    b2world = as_transform(pos_b_in_world, orn_b_in_world)

    pos_arm_in_world = np.array([0, 0, 0.5])
    orn_arm_in_world = R.from_euler('zyx', [0, 0, 0], degrees=True)
    world2arm = as_transform(pos_arm_in_world, orn_arm_in_world)

    calib_pt_arm = np.array([-323.30, -419.56, 550.15])
    calib_pt_cam = np.array([-394.38, -470, 467.00])
    pos_cam_in_arm = (calib_pt_arm - calib_pt_cam) / 1000
    orn_cam_in_arm = R.from_euler('zyx', [135, 0, 0], degrees=True)
    arm2cam = as_transform(pos_cam_in_arm, orn_cam_in_arm)

    tm = TransformManager()
    tm.add_transform("base", "world", b2world)
    tm.add_transform("arm", "world", world2arm)
    tm.add_transform("cam", "arm", arm2cam)


    pos_arm_in_world = np.array([-314.23, -514.34, 636.78]) / 1000
    orn_arm_in_world = R.from_euler('zxy', [-45.17, 0.04, 179.69], degrees=True)
    world2arm = as_transform(pos_arm_in_world, orn_arm_in_world)

    tm.add_transform("arm", "world", world2arm)

    ax = tm.plot_frames_in('world', s=0.1)
    ax.set_xlim((-0.7, 0.01))
    ax.set_ylim((-0.7, 0.01))
    ax.set_zlim((-0.01, 0.7))
    plt.show()
    exit(0)

    estimator = PoseEstimator(transform_manager=tm, window_size=args.averaging_window)
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
        
                pos, orn = estimator.estimate(frames)
                # pos = pos + np.array([0.055, 0, 0])
                # print(f'Pos {vec2str(pos)}')

                t = tm.get_transform('clamp', 'base')
                orn = R.from_matrix(t[:3, :3])

                t = tm.get_transform('clamp', 'world')
                pos = t[:3, -1] * 1000

                pos_desired = np.array([-318.6022, -395.1886, 17.7866])
                orn_desired = R.from_euler('zyx', [162.69, 0.32, -179.96], degrees=True)
                print(f'Pos world: {vec2str(pos)}')

                print(f'Orn world: {orn2str(orn)}')

                # t = tm.get_transform('arm', 'world')
                # orn = R.from_matrix(t[:3, :3])
                # pos = t[:3, -1] * 1000
                # print(f'Pos arm in world: {vec2str(pos)}')
                # print(f'Orn arm in world: {orn2str(orn)}')

                # t = tm.get_transform('cam', 'world')
                # orn = R.from_matrix(t[:3, :3])
                # pos = t[:3, -1] * 1000
                # print(f'Pos cam in world: {vec2str(pos)}')
                # print(f'Orn cam in world: {orn2str(orn)}')
            except Exception as e:
                print(f'Something went wrong! - {e}')
            print()

            cv.imshow('Display', estimator.create_img())
            key = cv.waitKey(1000 // 25)
            # key = cv.waitKey(0)
            if key == ord('p'):
                while key != ord('p'):
                    key = cv.waitKey(0)
                    if key == ord('q'):
                        exit(0)
            if key == ord('q'):
                break
            elif key == ord('p'):
                while (cv.waitKey(0) != ord('p')):
                    continue
    finally:
        pipeline.stop()
        cv.destroyAllWindows()

