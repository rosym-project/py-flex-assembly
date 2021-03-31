#!/usr/bin/env python3

import argparse
import math
import os
import sys

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

# import ros libraries
import roslib
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion

from gym_flexassembly.vision.pose_detection.projection.estimator import PoseEstimator, as_transform
from cosima_msgs.srv import GetClamp, GetClampResponse


class PoseService:

    def __init__(self, args):
        self.args = args
        self.tm = TransformManager()
        self.init_transfrom_manager()

        self.pose_estimator = PoseEstimator(debug=args.debug, window_size=args.averaging_window, transform_manager=self.tm)

        self.publisher = rospy.Publisher(args.topic, Pose, queue_size=10)
        self.service = rospy.Service(args.topic, GetClamp, self.handle_request)
        self.subscriber = rospy.Subscriber("/robot/fdb/cart_pose_0", Pose, self.update_arm_pose)

        self.pose = Pose()

        self.rs_config = rs.config()
        if args.file:
            self.rs_config.enable_device_from_file(args.file)

        if args.debug:
            cv.namedWindow('Display', cv.WINDOW_NORMAL)
            cv.resizeWindow('Display', 1600, 900)

    def init_transfrom_manager(self):
        pos_arm_in_world = np.array([0.0, 0.0, 1.0])
        orn_arm_in_world = R.from_quat([0.0, 0.0, 0.0, 1.0])
        world2arm = as_transform(pos_arm_in_world, orn_arm_in_world)
        self.tm.add_transform("arm", "world", world2arm)

        z = (550.15 - 467.00) / 1000
        calib_pt_arm = np.array([-333.72, -491.74, 373.78])
        calib_pt_cam = np.array([-407.27, -537.22, 539.82])
        pos_cam_in_arm = (calib_pt_arm - calib_pt_cam) / 1000
        pos_cam_in_arm = R.from_euler('zyx', [0, 180, 0], degrees=True).apply(pos_cam_in_arm)
        pos_cam_in_arm[2] = z
        orn_cam_in_arm = R.from_euler('zyx', [45, 0, 0], degrees=True)
        arm2cam = as_transform(pos_cam_in_arm, orn_cam_in_arm)
        self.tm.add_transform("cam", "arm", arm2cam)

    def handle_request(self, request):
        response = Pose()

        pos_w = self.tm.get_transform('clamp', 'world')[:3, -1]
        response.position.x = pos_w[0]
        response.position.y = pos_w[1]
        response.position.z = pos_w[2]

        orn_w = R.from_matrix(self.tm.get_transform('clamp', 'world')[:3, :3])
        # the gripper has to be rotated 90 degrees with regards to the clamp orientation
        orn_w = R.from_euler('zyx', [90, 0 ,0], degrees=True) * orn_w
        orn_wq = orn_w.as_quat()
        response.orientation.x = orn_wq[0]
        response.orientation.y = orn_wq[1]
        response.orientation.z = orn_wq[2]
        response.orientation.w = orn_wq[3]

        return GetClampResponse(response, None)

    def pose_to_transform(self, pose):
        orn = pose.orientation
        orn = [orn.w, orn.x, orn.y, orn.z]

        pos = pose.position
        pos = [pos.x, pos.y, pos.z]

        _pose = np.hstack((pos, orn))
        return pt.transform_from_pq(_pose)

    def update_arm_pose(self, pose):
        arm_transform = self.pose_to_transform(pose)
        self.tm.add_transform('arm', 'world', arm_transform)

    def run(self):
        self.rs_pipeline = rs.pipeline()
        self.rs_pipeline.start(self.rs_config)

        rate = rospy.Rate(2)
        try:
            while True:
                try:
                    frames = self.rs_pipeline.wait_for_frames()
                    pos, orn = self.pose_estimator.estimate(frames)
                    self.pose = self.pose_to_ros(pos, orn)

                    self.publisher.publish(self.pose)

                    if self.args.debug:
                        pos_w = self.tm.get_transform('clamp', 'world')[:3, -1] * 1000

                        orn_w = R.from_matrix(self.tm.get_transform('clamp', 'world')[:3, :3])
                        # the gripper has to be rotated 90 degrees with regards to the clamp orientation
                        orn_w = R.from_euler('zyx', [90, 0 ,0], degrees=True) * orn_w
                        orn_w = orn_w.as_euler('zyx', degrees=True)
                        print(f'Orn base: [{orn_w[0]:.2f}, {orn_w[1]:.2f}, {orn_w[2]:.3f}]')

                        cv.imshow('Display', self.pose_estimator.create_img())
                        if cv.waitKey(100) == ord('q'):
                            break
                    else:
                        rospy.sleep(0.1)
                except KeyboardInterrupt:
                    break 
                except Exception as e:
                    print(f'Pose estimation failed: {e}')
        finally:
            self.rs_pipeline.stop()
            if self.args.debug:
                cv.destroyAllWindows()

    def pose_to_ros(self, pos, orn):
        pose = Pose()
        pose.orientation = Quaternion(*orn.as_quat())
        pose.position = Point(*pos)
        return pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a ros service on the given topic to retrieve the pose of a detected clamp')
    parser.add_argument('--topic', type=str, default='pose_estimation',
                        help='the topic on which new poses are published')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--file', type=str)
    parser.add_argument('--averaging_window', type=int, default=25,
                        help='the detected pose is averaged over this many frames')
    args = parser.parse_args()
    print(args)

    # disable signals is needed to handle keyboard interrupt
    rospy.init_node('pose_estimator', anonymous=True, disable_signals=True)

    service = PoseService(args)
    service.run()
