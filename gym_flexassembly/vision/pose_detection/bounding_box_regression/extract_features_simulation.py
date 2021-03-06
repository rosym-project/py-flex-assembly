import argparse
import math
import sys

import numpy as np
import cv2 as cv
import tqdm


def detect_bounding_box(image, area_threshold):
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
            if stats[j, cv.CC_STAT_AREA] >= area_threshold:
                if clamp_found:
                    # already found a clamp => error
                    print("multiple clamps detected")
                    exit()
                # recompute the contour using only the current label
                contours, _ = cv.findContours(np.where(labels == j, 1, 0).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                clamp_found = True

        if not clamp_found:
            raise ValueError('Could not detect clamp')
    elif np.sum(mask) < area_threshold:
        raise ValueError('Could not detect clamp')

    # calculate an oriented bounding box
    return cv.minAreaRect(contours[0])


def detect_side(image, box):
    # compute laplacian
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    laplacian = np.where(laplacian > 0, 1, 0)

    # create masks for the two halves of the clamp
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

    # count pixels with laplacian > 0
    score_1 = np.sum(np.where(mask_1 == 1, laplacian, 0))
    score_2 = np.sum(np.where(mask_2 == 1, laplacian, 0))
    return 0 if score_1 > score_2 else 1


def detect_features(image, area_threshold=50):
    # ======================================================================
    # bounding box features
    # ======================================================================

    bounding_box = detect_bounding_box(image, area_threshold=area_threshold)
    box = cv.boxPoints(bounding_box)

    # extract the relevant data from the bounding box
    width = max(bounding_box[1])
    height = min(bounding_box[1])

    side_vec = []
    if np.linalg.norm(box[0] - box[1]) > np.linalg.norm(box[0] - box[3]):
        side_vec = box[1] - box[0]
    else:
        side_vec = box[3] - box[0]
    axis_vec = np.array([1, 0])
    angle = math.acos(axis_vec.dot(side_vec) / np.linalg.norm(side_vec))
    if np.linalg.norm(side_vec) < 0.1:
        print("warning: short side")

    # ======================================================================
    # additional features:
    # - which half has more pixels with laplacian > 0
    # - sum of laplacian
    # - number of pixels where laplacian > 0
    # ======================================================================

    laplacian_side = detect_side(image, box)

    # compute laplacian
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(gray, cv.CV_64F)

    # apply bounding box
    box_mask = np.zeros(laplacian.shape)
    cv.fillPoly(box_mask, [np.round(np.array(box)).astype(np.int)], color=1)
    laplacian = np.where(box_mask == 1, laplacian, 0)
    laplacian_sum = np.sum(laplacian)
    laplacian_abs_sum = np.sum(np.abs(laplacian))
    laplacian = np.where(laplacian != 0, 1, 0)
    laplacian_count = np.sum(laplacian)

    return box, [bounding_box[0][0], bounding_box[0][1], height, width, angle, height * width,
        laplacian_side, laplacian_sum, laplacian_abs_sum, laplacian_count]


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
    features = [["id", "image_name", "bb_center_x", "bb_center_y", "bb_height", "bb_width",
                "bb_angle[radians]", "bb_area", "laplacian_side", "laplacian_sum", "laplacian_abs_sum", "laplacian_count"]]


    for i, f in enumerate(tqdm.tqdm(data)):
        image = cv.imread(args.data_dir + "/" + f)

        try:
            box, feature_vec = detect_features(image, area_threshold=args.area_threshold)
        except ValueError as e:
            print(e)
            exit()

        # ======================================================================
        # save feature + visualization
        # ======================================================================

        # append feature vector to list
        features.append([i, f] + feature_vec)

        # visualization
        if args.visualize:
            # draw bounding box
            box = np.intp(box)
            cv.drawContours(image, [box], 0, (0,255,0))

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

            if feature_vec[6] == 0: # laplacian_side
                cv.polylines(image, [half_1], True, color=(0,0,255))
            else:
                cv.polylines(image, [half_2], True, color=(0,0,255))

            cv.imshow("image", image)

            #cv.drawContours(mask, [box], 0, 125)
            #cv.imshow("mask", mask)

            if cv.waitKey(0) == ord('q'):
                break

    cv.destroyAllWindows()
    np.savetxt(args.data_dir + "/features.csv", features, fmt='%s', delimiter=',')


if __name__ == '__main__':
    main(sys.argv)
