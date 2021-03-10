import argparse
import math
import sys

import numpy as np
import cv2 as cv
import tqdm

def discriminant_analysis(img):
    """
    Perform a discriminant analysis to find an optimal threshold
    for an image.
    """
    hist = cv.calcHist([img], [0], None, [256], [0, 256], accumulate=False)
    hist = hist[:, 0]
    hist = hist / (img.shape[0] * img.shape[1])

    threshes = np.arange(0, 256)
    mean_t = np.sum(threshes * hist)

    values = []
    for t in threshes:
        mean_1 = np.sum(threshes[:t] * hist[:t])
        mean_2 = np.sum(threshes[t:] * hist[t:])

        var_i_1 = np.sum(hist[:t] * np.power(threshes[:t] - mean_1, 2))
        var_i_2 = np.sum(hist[t:] * np.power(threshes[t:] - mean_2, 2))
        var_i = var_i_1 + var_i_2

        var_b_1 = np.power(mean_1 - mean_t, 2) * np.sum(hist[:t])
        var_b_2 = np.power(mean_2 - mean_t, 2) * np.sum(hist[t:])
        var_b = var_b_1 + var_b_2

        values.append(var_b / var_i)
    return np.argmax(values)


def detect_bb(depth, area_threshold):
    """
    Compute a bounding box for a clamp based on a thresholded depth image.
    The threshold is computed with a discriminant analysis.
    """
    # ignore the border of the depth images since they have a low certainty
    # and thus contain large outliers
    #TODO test if this is still needed if the depth image is averaged over multiple frames
    offset_width = 100
    offset_height = 100
    _depth = depth[offset_height:depth.shape[0] - offset_height, \
                   offset_width:depth.shape[1] - offset_width]
    _depth = cv.convertScaleAbs(_depth, alpha=0.3)

    # compute and apply a threshold
    threshold = discriminant_analysis(_depth)
    _, thresholded = cv.threshold(_depth, threshold, 255, cv.THRESH_BINARY_INV)

    # perform closing for holes
    thresholded = cv.dilate(thresholded, np.ones((5, 5), np.uint8), iterations=10)
    thresholded = cv.erode(thresholded, np.ones((5, 5), np.uint8), iterations=10)

    #cv.imshow('threshold', thresholded)

    # find all contours on the thresholded image
    cnts, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE, offset=(offset_width, offset_height))
    # filter contours whose area is too small
    cnts = list(filter(lambda cnt: cv.contourArea(cnt) > area_threshold, cnts))
    # filter areas that are too large (outside the border of the table)
    def too_large_filter(cnt):
        _, _, width, height = cv.boundingRect(cnt)
        if width > _depth.shape[1] / 2 or height > _depth.shape[0] / 2:
            return False
        return True
    cnts = list(filter(too_large_filter, cnts))

    # sort regions by their average depth value to find the region containing
    # the clamp since it should have the lowest distance from the camere
    def avg_elevation_key(cnt):
        mask = np.zeros(depth.shape, np.uint8)
        cv.drawContours(mask, [cnt], -1, 255, -1,)
        elevation_sum = np.where(mask == 255, depth, np.zeros(depth.shape)).sum()
        area = cv.contourArea(cnt)
        return elevation_sum / area
    cnts.sort(key=avg_elevation_key)

    return cv.minAreaRect(cnts[0])


def refine_bb(bounding_box, image):
    """
    Refines an existing bounding box by executing the following steps:
    - Enlarge the bounding box to include the border of the clamp that might have been missed
    - Apply a high pass filter to the image emphasize the edges of the clamp
    - Use grabCut to extract the clamp from the image.
      The resulting bounding box is a lot better than the initial bounding box from the depth image.
    """

    # enlarge the bounding box
    size_increase = 1.2
    new_size = tuple([bounding_box[1][0] * size_increase, bounding_box[1][1] * size_increase])
    new_bb = tuple([bounding_box[0], new_size, bounding_box[2]])

    # create the mask
    mask = np.zeros(image.shape[:2], np.uint8)
    box = cv.boxPoints(new_bb)
    box = np.intp(box)
    cv.drawContours(mask, [box], 0, cv.GC_PR_FGD, -1)

    # only work on a cutout to improve runtime
    rect = cv.boundingRect(box)
    shape = np.array([rect[2], rect[3]])
    p_1 = np.array([rect[0], rect[1]])
    p_2 = p_1 + shape

    # apply a high pass filter to emphasize the edges
    kernel_size = 21
    kernel = cv.getGaussianKernel(kernel_size, 0)
    kernel = -1 * np.outer(kernel, kernel)
    kernel[int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)] += 2
    filtered = cv.filter2D(image[p_1[1] : p_2[1], p_1[0] : p_2[0]], cv.CV_8UC1, kernel)
    #cv.imshow("filter", filtered)

    # use grabCut to refine the mask borders
    bgdModel = np.zeros((1, 65), np.float64) # 13*components_count rows
    fgdModel = np.zeros((1 ,65), np.float64)
    cv.grabCut(filtered, mask[p_1[1] : p_2[1], p_1[0] : p_2[0]], None, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK)
    mask = np.isin(mask, [cv.GC_FGD, cv.GC_PR_FGD]).astype(np.uint8)

    # debug: show grabCut results
    """
    clamp = np.copy(image)
    clamp[np.where(mask==1, False, True)] = [0, 0, 0]
    cv.imshow("mask, grabCut", mask * 255)
    cv.imshow("image, grabCut", clamp)
    """
    #"""

    # detect a contour around the clamp
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        print("Could not detect clamp. Aborting bounding box refinement.")
        return bounding_box
    elif len(contours) > 1:
        # grabCut often splits the clamp into multiple parts
        # => compute convex hull of contours
        contours = np.concatenate(contours)
        contours = contours.reshape(-1, contours.shape[-1]).astype(np.float32)
        contours = cv.convexHull(contours)
        contours = [contours]

    # calculate an oriented bounding box
    return cv.minAreaRect(contours[0])


def detect_side(image, box):
    """
    Uses a Hough Circle Transform and post processing of the detected circles
    to detect the side of the clamp that has the hole in it.
    """
    # only work on a cutout to improve runtime
    rect = cv.boundingRect(box)
    shape = np.array([rect[2], rect[3]])
    p_1 = np.array([rect[0], rect[1]])
    p_2 = p_1 + shape

    # detect circles
    gray = cv.cvtColor(image[p_1[1] : p_2[1], p_1[0] : p_2[0]], cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 3)
    #cv.imshow("gray", gray)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 10,
                            param1=50, param2=10, minRadius=4, maxRadius=6)
    if circles is None:
        # no regions detected, no prediction possible
        # since this occurs rarely it is valid to predict a fixed side
        raise RuntimeError('Could not detect holes')
    circles = circles[0]

    # adjust circle translation
    for c in circles:
        c[:2] += p_1

    # create boxes for the regions where the hole is expected to be
    if np.linalg.norm(box[0] - box[1]) < np.linalg.norm(box[1] - box[2]):
        side_vec = box[2] - box[1]
        b_1 = [box[0], box[1]]
        b_2 = [box[3], box[2]]
    else:
        side_vec = box[0] - box[1]
        b_1 = [box[1], box[2]]
        b_2 = [box[0], box[3]]

    lower = 0.25
    upper = 0.425

    b_1 = [b_1[0] + lower * side_vec, b_1[1] + lower * side_vec,
           b_1[1] + upper * side_vec, b_1[0] + upper * side_vec]
    b_1 = np.array(b_1)
    b_2 = [b_2[0] - lower * side_vec, b_2[1] - lower * side_vec,
           b_2[1] - upper * side_vec, b_2[0] - upper * side_vec]
    b_2 = np.array(b_2)

    #cv.polylines(image, [np.round(b_1).astype(np.int)], True, color=(255,0,0))
    #cv.polylines(image, [np.round(b_2).astype(np.int)], True, color=(255,0,0))

    # filter the detected circles
    # count the number of circles in each region and average their color value
    count_1, count_2 = 0, 0
    color_1, color_2 = np.zeros(3, np.uint64), np.zeros(3, np.uint64)
    pixel_1, pixel_2 = 0, 0

    for i in range(circles.shape[0]):
        if cv.pointPolygonTest(b_1, tuple(circles[i, 0:2]), False) == 1:
            count_1 += 1

            mask = np.zeros(image.shape, np.uint8)
            c = np.uint16(np.around(circles[i]))
            center = (c[0], c[1])
            radius = c[2]
            cv.circle(mask, center, radius, (1, 1, 1), -1)
            color_1 += np.sum(np.multiply(mask, image), axis=tuple([0, 1]))
            pixel_1 += np.sum(mask)

        elif cv.pointPolygonTest(b_2, tuple(circles[i, 0:2]), False) == 1:
            count_2 += 1

            mask = np.zeros(image.shape, np.uint8)
            c = np.uint16(np.around(circles[i]))
            center = (c[0], c[1])
            radius = c[2]
            cv.circle(mask, center, radius, (1, 1, 1), -1)
            color_2 += np.sum(np.multiply(mask, image), axis=tuple([0, 1]))
            pixel_2 += np.sum(mask)

        else:
            circles[i, :] = np.full((3), np.nan)

    # remove circles outside the regions
    circles = circles[~np.isnan(circles).any(axis=1)]

    # average the colors if both regions contain circles
    if pixel_1 > 0 and pixel_2 > 0:
        color_1 = color_1.astype(np.float)
        color_1 /= pixel_1
        color_2 = color_2.astype(np.float)
        color_2 /= pixel_2

    # draw circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles:
            center = (i[0], i[1])
            # circle outline
            radius = i[2]
            cv.circle(image, center, radius, (0, 0, 255), 1)

    if count_1 > count_2:
        return 0
    elif count_2 > count_1:
        return 1
    else:
        # same number of circles in both regions
        if count_1 == 0 and count_2 == 0:
            # no regions detected, no prediction possible
            # since this occurs rarely it is valid to predict a fixed side
            # print("No holes detected")
            # return 0
            raise RuntimeError('No holes detected')
        elif np.linalg.norm(color_1) < np.linalg.norm(color_2):
            # the darker region is more likely to contain the hole
            return 0
        else:
            return 1


def detect_features(image, depth, area_threshold=50):
    """
    Extracts a bounding box and a feature vector from a clamp image and its depth image.
    """

    # ======================================================================
    # bounding box features
    # ======================================================================

    bounding_box = detect_bb(depth, area_threshold=area_threshold)
    bounding_box = refine_bb(bounding_box, image)
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
    # - which half is the one with the hole in it
    # - sum of laplacian
    # - number of pixels where laplacian > 0
    # ======================================================================

    side = detect_side(image, box)

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
        side, laplacian_sum, laplacian_abs_sum, laplacian_count]


def visualize_features(image, depth, box, feature_vec):
    """
    Visualizes the extracted bounding box.
    """
    # visualize_features the depth image
    depth = cv.convertScaleAbs(depth, alpha=0.3)
    depth = cv.equalizeHist(depth)
    depth = cv.applyColorMap(depth, cv.COLORMAP_JET)
    #cv.imshow("depth", depth)

    # draw bounding box
    box = np.intp(box)
    cv.drawContours(image, [box], 0, (0,255,0))

    # draw a box around the side containing the hole
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

    # cv.imshow("image", image)
    # return cv.waitKey(0)
    return image


def main(args):
    parser = argparse.ArgumentParser('Calculates the bounding boxes and other features for an image dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str, default='./real/position_01',
                        help='the directory of the dataset')
    parser.add_argument('--area_threshold', type=int, default=50,
                        help='the minimal size of a clamp')
    parser.add_argument('-v', '--visualize_features', action="store_true",
                        help='visualize_features the clamp detection')
    args = parser.parse_args(args[1:])
    print(args)

    image_data = np.loadtxt(args.data_dir + "/data.csv", skiprows=1, usecols=1, delimiter=',', dtype=str)
    depth_data = np.loadtxt(args.data_dir + "/data.csv", skiprows=1, usecols=2, delimiter=',', dtype=str)
    features = [["id", "image_name", "bb_center_x", "bb_center_y", "bb_height", "bb_width",
                "bb_angle[radians]", "bb_area", "laplacian_side", "laplacian_sum", "laplacian_abs_sum", "laplacian_count"]]


    for i in tqdm.tqdm(range(len(image_data))):
        # load the images
        image = cv.imread(args.data_dir + "/" + image_data[i])
        depth = np.load(args.data_dir + "/" + depth_data[i])

        # extract the features
        try:
            box, feature_vec = detect_features(image, depth, area_threshold=args.area_threshold)
        except ValueError as e:
            print(e)
            exit()

        # visualize_features the results
        if args.visualize_features:
            cv.imshow('image', visualize_features(image, depth, box, feature_vec))
            key = cv.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('c'):
                print(f'Corrected the detected side for image: {image_data[i]}')
                feature_vec[6] = 0 if feature_vec[6] == 1 else 1
            elif key == ord('s'):
                print(f'Skip image: {image_data[i]}')
                continue

        # append feature vector to list
        features.append([i, image_data[i]] + feature_vec)

    cv.destroyAllWindows()
    np.savetxt(args.data_dir + "/features.csv", features, fmt='%s', delimiter=',')


if __name__ == '__main__':
    main(sys.argv)
