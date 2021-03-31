import argparse
from dataclasses import dataclass
import os

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

import gym_flexassembly.vision.pose_detection.projection.features as fts
import gym_flexassembly.vision.pose_detection.projection.visualize as vis
from gym_flexassembly.vision.pose_detection.projection.estimator import FilterList


def img_from_half(img_color, half):
    ys = half[:, 0]
    xs = half[:, 1]
    ys = (ys.min(), ys.max())
    xs = (xs.min(), xs.max())
    h = ys[1] - ys[0]
    w = xs[1] - xs[0]

    if h < w:
        diff = (w - h) // 2
        ys = (ys[0] - diff, ys[1] + diff)
    else:
        diff = (h - w) // 2
        xs = (xs[0] - diff, xs[1] + diff)

    img = img_color[xs[0]:xs[1], ys[0]:ys[1]]
    return cv.resize(img, (224, 224))


@dataclass
class HalfHolder:
    half_hole: np.array
    half_other: np.array

    def switch(self):
        self.half_hole, self.half_other = self.half_other, self.half_hole

    def show(self, img):
        return img_from_half(img, self.half_hole), img_from_half(img, self.half_other)


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
parser.add_argument('-s', '--skip', type=int, default=10)
parser.add_argument('--id', type=int, default=0)
parser.add_argument('--dir', type=str, default='./dataset/side')
args = parser.parse_args()

hole_dir = os.path.join(args.dir, 'hole')
other_dir = os.path.join(args.dir, 'other')
try:
    os.mkdir(hole_dir)
except FileExistsError:
    pass
try:
    os.mkdir(other_dir)
except FileExistsError:
    pass

config = rs.config()
config.enable_device_from_file(args.file)

pipeline = rs.pipeline()
pipeline.start(config)

align = rs.align(rs.stream.color)
f = FilterList()

current_id = args.id

while True:
    for i in range(args.skip):
        frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    frame_color = frames.get_color_frame()
    frame_depth = f.process(frames.get_depth_frame())

    # retrieve color and depth images
    img_color = np.asanyarray(frame_color.get_data())
    img_color = cv.cvtColor(img_color, cv.COLOR_RGB2BGR)
    img_depth = np.asanyarray(frame_depth.get_data())

    bb_vis = np.zeros((*img_depth.shape, 3), np.uint8)

    bb = fts.detect_bounding_box(fts.flatten(img_depth), viz=bb_vis)
    scale_h = img_color.shape[1] / img_depth.shape[1]
    scale_w = img_color.shape[0] / img_depth.shape[0]
    bb.scale(scale_h, scale_w)
    bb = fts.refine_bb(bb, img_color)
    try:
        side = fts.detect_side(img_color, bb.as_points(), visualize=False)
    except RuntimeError:
        print('Could not detect side')
        side = 0

    box = bb.as_int()
    if np.linalg.norm(box[0] - box[1]) < np.linalg.norm(box[1] - box[2]):
        half_1 = [box[0], box[1]]
        half_2 = [box[3], box[2]]
    else:
        half_1 = [box[1], box[2]]
        half_2 = [box[0], box[3]]
    middle = [(half_1[0] + half_2[0]) / 2, (half_1[1] + half_2[1]) / 2]
    middle.reverse()
    half_1.extend(middle)
    half_1 = np.round(np.array(half_1)).astype(np.int)
    half_2.extend(middle)
    half_2 = np.round(np.array(half_2)).astype(np.int)

    halves = [half_1, half_2]

    if side == 0:
        holder = HalfHolder(*halves)
    else:
        holder = HalfHolder(*halves[::-1])



    empty = np.zeros(img_color.shape, np.uint8)
    img_depth = cv.resize(vis.display_depth_image(img_depth), img_color.shape[:2][::-1])
    bb_vis = cv.resize(bb_vis, img_color.shape[:2][::-1])
    left = np.hstack((img_color, empty))
    right = np.hstack((img_depth, bb_vis))
    cv.namedWindow('Display', cv.WINDOW_NORMAL)
    cv.resizeWindow('Display', 1600, 900)
    cv.imshow('Display', np.vstack((left, right)))

    img_hole, img_other = holder.show(img_color)
    cv.imshow('No hole', img_other)
    cv.imshow('Hole', img_hole)
    key = cv.waitKey(0)
    while key != ord('n'):
        if key == ord('q'):
            exit(0)
        if key == ord('c'):
            print('Switch')
            holder.switch()

            img_hole, img_other = holder.show(img_color)
            cv.imshow('No hole', img_other)
            cv.imshow('Hole', img_hole)
        if key == ord('s'):
            filename = f'{current_id:04d}.png'
            print(f'Save as {filename}')
            
            current_id += 1

            cv.imwrite(os.path.join(hole_dir, filename), img_hole)
            cv.imwrite(os.path.join(other_dir, filename), img_other)
            break


        key = cv.waitKey(0)

