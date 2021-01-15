import argparse
import math
import sys

import cv2 as cv
import numpy as np
import tqdm

def main(args):
    parser = argparse.ArgumentParser('Calculates the bounding boxes and other features for an image dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str, default='./low_var_dataset/low_var/train',
                        help='the directory of the dataset')
    args = parser.parse_args(args[1:])
    print(args)

    data = np.loadtxt(args.data_dir + "/data.csv", skiprows=1, usecols=1, delimiter=',', dtype=np.str)

    for f in tqdm.tqdm(data):
        image = cv.imread(args.data_dir + "/" + f)

        # create a mask containing the clamp
        # since the background is a simple gray it is easiest to detect the background and invert the resulting mask
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower = np.array([0, 0, 20])
        upper = np.array([255, 10, 255])
        mask = cv.bitwise_not(cv.inRange(image, lower, upper))

        # detect a contour around the clamp
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(contours) != 1:
            # todo: only keep largest area
            print("multiple contours detected")
            exit()

        # calculate an oriented bounding box
        bounding_box = cv.minAreaRect(contours[0])

        # visualization
        image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
        # draw bounding box
        box = cv.boxPoints(bounding_box)
        box = np.intp(box)
        cv.drawContours(image, [box], 0, (0,255,0))
        cv.drawContours(mask, [box], 0, 125)

        cv.imshow("image", image)
        cv.imshow("mask", mask)
        if cv.waitKey(0) == ord('q'):
            break

if __name__ == '__main__':
    main(sys.argv)
