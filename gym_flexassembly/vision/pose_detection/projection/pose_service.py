#!/usr/bin/env python3

import argparse
import math
import os
import sys

import cv2 as cv
import pyrealsense2 as rs

# import ros libraries
import roslib
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion

from gym_flexassembly.vision.pose_detection.projection.estimator import PoseEstimator
from py_flex_assembly.srv import PoseEstimation, PoseEstimationResponse


class PoseService:

    def __init__(self, args):
        self.args = args
        self.pose_estimator = PoseEstimator(debug=args.debug, window_size=args.averaging_window)

        self.publisher = rospy.Publisher(args.topic, Pose, queue_size=10)
        self.service = rospy.Service(args.topic, PoseEstimation, self.handle_request)

        self.pose = Pose()

        self.rs_config = rs.config()
        if args.file:
            self.rs_config.enable_device_from_file(args.file)

        if args.debug:
            cv.namedWindow('Display', cv.WINDOW_NORMAL)
            cv.resizeWindow('Display', 1600, 900)


    def handle_request(self, request):
        return PoseEstimationResponse(self.pose)

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
                        cv.imshow('Display', self.pose_estimator.create_img())
                        if cv.waitKey(10) == ord('q'):
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
