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

from gym_flexassembly.vision.pose_estimation.estimator import PoseEstimator

# TODO:
# * Verbindung zu Realsense (auch über File)
# * Pose berechnen (im Vgl. zu Arm?)
# * Pos und Orn mittel (über Parameter)
# * Error Handling

# * Berechnungszeit messen
# * Überlegen ob Bilder über andere Komponente gegrabbed werden sollten...


class PoseService:

    def __init__(self, args):
        self.args = args
        self.pose_estimator = PoseEstimator(debug=args.debug)

        self.publisher = rospy.Publisher(args.topic, Pose, queue_size=10)
        #self.service = rospy.Service('pose_estimations', Pose, self.handle_request)

        self.pose = Pose()

        self.rs_config = rs.config()
        if args.file:
            self.rs_config.enable_device_from_file(args.file)

        if args.debug:
            cv.namedWindow('Display', cv.WINDOW_NORMAL)
            cv.resizeWindow('Display', 1600, 900)


    def handle_request(self, request):
        return self.pose

    def run(self):
        self.rs_pipeline = rs.pipeline()
        self.rs_pipeline.start(self.rs_config)

        rate = rospy.Rate(100)
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
                        rate.sleep()
                except KeyboardInterrupt:
                    print('KeyboardInterrupt...')
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
    #TODO
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('--topic', type=str, default='pose_estimation',
                        help='the topic on which new poses are published')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--file', type=str)
    # parser.add_argument('--image', type=str, required=True,
                        # help='the image file to be proessed')
    # parser.add_argument('--data', type=str,
                        # help='the data csv file (if available)')
    args = parser.parse_args()
    print(args)

    rospy.init_node('pose_estimator', anonymous=True)

    service = PoseService(args)
    service.run()
