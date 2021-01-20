import argparse
import math
import sys

import numpy as np
import cv2 as cv
import tqdm

def main(args):
    parser = argparse.ArgumentParser('Calculates the bounding boxes and other features for an image dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str, default='./low_var_dataset/low_var/train',
                        help='the directory of the dataset')
    parser.add_argument('--area_threshold', type=int, default=50,
                        help='the minimal size of a clamp')
    parser.add_argument('-v', '--visualize', action="store_true",
                        help='visualize the clamp detection')
    args = parser.parse_args(args[1:])
    print(args)

    data = np.loadtxt(args.data_dir + "/data.csv", skiprows=1, usecols=1, delimiter=',', dtype=np.str)
    features = [["id", "image_name", "bb_center_x", "bb_center_y", "bb_height", "bb_width", "bb_angle"]]

    for i, f in enumerate(tqdm.tqdm(data)):
        image = cv.imread(args.data_dir + "/" + f)

        # ======================================================================
        # bounding box features
        # ======================================================================

        # create a mask containing the clamp
        # since the background is a simple gray it is easiest to detect the background and invert the resulting mask
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower = np.array([0, 0, 20])
        upper = np.array([255, 10, 255])
        mask = cv.bitwise_not(cv.inRange(image, lower, upper))
        image = cv.cvtColor(image, cv.COLOR_HSV2BGR)

        # detect a contour around the clamp
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(contours) != 1:
            # multiple countours detected => ignore small areas
            retval, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=4)

            clamp_found = False
            # iterate through all areas (ignore label 0 = background)
            for j in range(1, retval):
                if stats[j, cv.CC_STAT_AREA] >= args.area_threshold:
                    if clamp_found:
                        # already found a clamp => error
                        print("multiple clamps detected")
                        exit()
                    # recompute the contour using only the current label
                    countours, _ = cv.findContours(np.where(labels == j, 1, 0).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                    clamp_found = True

            if not clamp_found:
                print("no clamps detected")
                exit()
        elif np.sum(mask) < args.area_threshold:
            # clamp area too small
            print("no clamps detected")
            exit()

        # calculate an oriented bounding box
        bounding_box = cv.minAreaRect(contours[0])


        # ======================================================================
        # additional features
        # ======================================================================

        # detect hole
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #laplacian = cv.Laplacian(gray, cv.CV_8U)
        #laplacian = np.where(laplacian > 0, 255, 0).astype(np.uint8)
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 10,
                               param1=25, param2=15,
                               minRadius=5, maxRadius=15)

        # gradient detection
        laplacian = cv.Laplacian(gray, cv.CV_8U)
        sobelx = cv.Sobel(gray, cv.CV_64F,1,0,ksize=3)
        sobely = cv.Sobel(gray, cv.CV_64F,0,1,ksize=3)
        #cv.imshow("laplacian", np.minimum(laplacian.astype(np.float64) * 255, 255))
        #cv.imshow("sobelx", sobelx)
        #cv.imshow("sobely", sobely)
        #if cv.waitKey(0) == ord('q'):
            #break

        # append results to feature vector
        features.append([i, f, bounding_box[0][0], bounding_box[0][1], bounding_box[1][0], bounding_box[1][1], bounding_box[2]])

        # visualization
        if args.visualize:
            # draw bounding box
            box = cv.boxPoints(bounding_box)
            box = np.intp(box)
            cv.drawContours(image, [box], 0, (0,255,0))
            #cv.drawContours(mask, [box], 0, 125)

            # draw circles
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    # circle outline
                    radius = i[2]
                    cv.circle(image, center, radius, (0, 0, 255), 1)

            cv.imshow("image", image)
            #cv.imshow("mask", mask)
            if cv.waitKey(0) == ord('q'):
                break

    cv.destroyAllWindows()
    np.savetxt(args.data_dir + "/features.csv", features, fmt='%s', delimiter=',')
if __name__ == '__main__':
    main(sys.argv)
