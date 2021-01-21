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
        # doesn't work reliably
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #laplacian = cv.Laplacian(gray, cv.CV_8U)
        #laplacian = np.where(laplacian > 0, 255, 0).astype(np.uint8)
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 10,
                               param1=25, param2=15,
                               minRadius=5, maxRadius=15)

        # gradient detection
        laplacian = cv.Laplacian(gray, cv.CV_64F)
        laplacian = np.where(laplacian > 0, 1, 0)
        #cv.imshow("laplacian", np.minimum(laplacian.astype(np.float64) * 255, 255))
        #if cv.waitKey(0) == ord('q'):
            #break

        # create masks for the two halves of the clamp
        box = cv.boxPoints(bounding_box)
        half_1 = []
        half_2 = []
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

        mask_1 = np.zeros(image.shape[:2])
        cv.fillPoly(mask_1, [half_1], color=1)
        mask_2 = np.zeros(image.shape[:2])
        cv.fillPoly(mask_2, [half_2], color=1)

        score_1 = np.sum(np.where(mask_1 == 1, laplacian, 0))
        score_2 = np.sum(np.where(mask_2 == 1, laplacian, 0))

        if score_1 > score_2:
            cv.polylines(image, [half_1], True, color=(0,0,255))
        else:
            cv.polylines(image, [half_2], True, color=(0,0,255))

        #cv.polylines(image, [half_1], True, color=(255,0,0))
        #cv.polylines(image, [half_2], True, color=(0,0,255))

        # append feature vector to list
        features.append([i, f, bounding_box[0][0], bounding_box[0][1], bounding_box[1][0], bounding_box[1][1], bounding_box[2]])

        # visualization
        if args.visualize:
            # draw bounding box
            box = cv.boxPoints(bounding_box)
            box = np.intp(box)
            #cv.drawContours(image, [box], 0, (0,255,0))
            cv.drawContours(mask, [box], 0, 125)

            """
            cv.putText(image, '0', tuple(box[0]), 0, 0.4, (0, 0, 0))
            cv.putText(image, '1', tuple(box[1]), 0, 0.4, (0, 0, 0))
            cv.putText(image, '2', tuple(box[2]), 0, 0.4, (0, 0, 0))
            cv.putText(image, '3', tuple(box[3]), 0, 0.4, (0, 0, 0))
            """

            # draw circles
            """
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    # circle outline
                    radius = i[2]
                    cv.circle(image, center, radius, (0, 0, 255), 1)
            """

            cv.imshow("image", image)
            #cv.imshow("mask", mask)
            if cv.waitKey(0) == ord('q'):
                break

    cv.destroyAllWindows()
    np.savetxt(args.data_dir + "/features.csv", features, fmt='%s', delimiter=',')
if __name__ == '__main__':
    main(sys.argv)
