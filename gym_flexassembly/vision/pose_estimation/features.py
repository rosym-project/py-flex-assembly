import cv2 as cv
import numpy as np

from sklearn import linear_model


class BoundingBox:

    def __init__(self, rotated_rect):
        self.pt = np.array(rotated_rect[0])
        self.w, self.h = rotated_rect[1]
        self.angle = rotated_rect[2]

    def as_rotated_rect(self):
        return ((tuple(self.pt), (self.w, self.h), self.angle))

    def as_points(self):
        return cv.boxPoints(self.as_rotated_rect())

    def as_int(self):
        return self.as_points().astype(np.int) 

    def scale(self, scale_h, scale_w, move_center=True):
        self.w *= scale_w
        self.h *= scale_h

        if move_center:
            self.pt[0] *= scale_h
            self.pt[1] *= scale_w

        return self

    def copy(self):
        return BoundingBox(self.as_rotated_rect())

    def as_slice(self):
        _pts = self.as_int()
        return slice(min(_pts[:, 1]), max(_pts[:, 1])), slice(min(_pts[:, 0]), max(_pts[:, 0]))

    def __repr__(self):
        return f'{self.as_rotated_rect()}'


def compute_aspect_ratio(rotated_rect):
    """
    Compute the aspect ratio of a rotated rectangle as the
    ratio between length of the longer and the shorter side.
    """
    width, height = rotated_rect[1]
    return max(width, height) / min(width, height)


def compute_rectangleness(contour):
    """
    Compute a measure for how much a contour resembles a
    rectangle.
    """
    contoure_area = cv.contourArea(contour)
    _, (w, h), _ = cv.minAreaRect(contour)
    return contoure_area / (w * h)


def flatten(img_depth):
    """
    Flatten a depth image by fitting a plane to the depth image (should be the
    table). A difference of a constant depth to the plane is added to flatten
    the data.

    In addition all zero value are set to the maximum depth. This makes the bounding box detection more robust since the clamp is most likely the closest element
    in the image.
    """
    xs, ys = np.nonzero(img_depth)
    # only use each eighth value for faster computation
    xs = xs[::8]
    ys = ys[::8]
    depths = img_depth[xs, ys]

    # fit plane
    inputs = np.array([xs, ys]).T
    model = linear_model.LinearRegression().fit(inputs, depths)

    # create plane image
    xx = np.arange(img_depth.shape[1])[::-1]
    yy = np.arange(img_depth.shape[0])[::-1]
    xx, yy = np.meshgrid(xx, yy)
    plane = model.intercept_ + model.coef_[1] * xx + model.coef_[0] * yy

    depths = plane.ravel()
    args = np.argsort(depths)
    const_depth = depths[args[args.shape[0] // 20]]

    max_depth = np.max(img_depth)
    res = np.where(img_depth > 0, img_depth + (plane - const_depth), max_depth)
    return res


def detect_bounding_box(depth, min_area=200, viz=None):
    """
    Detect a bounding box around a clamp (the nearest area in the depth image).
    """
    # cut-off the borders of the dept image since they contain error-prone regions and high noise
    offset_width = depth.shape[1] // 6
    offset_height = depth.shape[0] // 6
    _depth = depth[offset_height:depth.shape[0] - offset_height, \
                   offset_width:depth.shape[1] - offset_width]
    _depth = (_depth - _depth.min()) / (_depth.max() - _depth.min())
    _depth = (255 * _depth).astype(np.uint8)

    # equalize hist should make that the nearest area (the clamp) have the lowest value 
    _depth = cv.equalizeHist(_depth)
    _, thresholded = cv.threshold(_depth, 2, 255, cv.THRESH_BINARY_INV)

    thresholded = cv.dilate(thresholded, np.ones((3, 3)), iterations=3)
    thresholded = cv.erode(thresholded, np.ones((3, 3)), iterations=3)

    # find contours in threshold image
    cnts, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE, offset=(offset_width, offset_height))
    # filter contours with represent a too small area for  a clamp
    cnts = list(filter(lambda cnt: cv.contourArea(cnt) >= min_area, cnts))
    # filter contours which do not represent rectangular regions
    #TODO: check if should be used or not
    # cnts = list(filter(lambda cnt: compute_rectangleness(cnt) > 0.50, cnts))
    # compute rectangles for contours
    rects = list(map(lambda cnt: cv.minAreaRect(cnt), cnts))
    # sort contours by their aspect ration
    rects.sort(key=lambda rect: abs(compute_aspect_ratio(rect) - 7.5))
    rects = rects[::-1]

    #print('Rectness', list(map(lambda r: compute_rectangleness(r), cnts)))

    if viz is not None:
        _h = offset_height
        _w = offset_width
        _thresholded = cv.copyMakeBorder(thresholded, _h, _h, _w, _w, cv.BORDER_CONSTANT, 0)
        _thresholded = _thresholded.repeat(3).reshape(viz.shape)

        color_from = np.array([255, 0, 0])
        color_to = np.array([0, 255, 0])
        for i, rect in enumerate(rects):
            w = 1.0 if len(rects) == 1 else i / (len(rects) - 1)
            color = (1.0 - w) * color_from + w * color_to
            color = color.astype(np.int).tolist()
            cv.drawContours(_thresholded, [cv.boxPoints(rect).astype(np.int)], -1, color, 3)

        np.copyto(viz, _thresholded)

    return BoundingBox(rects[-1])


def refine_bb(bounding_box, image):
    """
    Refines an existing bounding box by executing the following steps:
    - Enlarge the bounding box to include the border of the clamp that might have been missed
    - Apply a high pass filter to the image emphasize the edges of the clamp
    - Use grabCut to extract the clamp from the image.
      The resulting bounding box is a lot better than the initial bounding box from the depth image.
    """
    # enlarge the bounding box
    bounding_box = bounding_box.copy().scale(1.2, 1.2, move_center=False)

    # create the mask
    mask = np.full(image.shape[:2], cv.GC_BGD, np.uint8)
    cv.drawContours(mask, [bounding_box.as_int()], 0, cv.GC_PR_FGD, -1)

    # only work on a cutout to improve runtime
    bounding_box.scale(1.2, 1.2, move_center=False)
    _img = image[bounding_box.as_slice()]
    _mask = mask[bounding_box.as_slice()]

    # apply a high pass filter to emphasize the edges
    kernel_size = 21
    kernel = cv.getGaussianKernel(kernel_size, 0)
    kernel = -1 * np.outer(kernel, kernel)
    kernel[int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)] += 2
    filtered = cv.filter2D(_img, cv.CV_8UC1, kernel)

    # use grabCut to refine the mask borders
    bgdModel = np.zeros((1, 65), np.float64) # 13*components_count rows
    fgdModel = np.zeros((1 ,65), np.float64)
    cv.grabCut(filtered, _mask, None, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK)
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
    return BoundingBox(cv.minAreaRect(contours[0]))


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
            cv.circle(image, center, radius, (0, 0, 255), 2)

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


def visualize_features(image, bounding_box, side):
    """
    Visualizes the extracted bounding box.
    """
    # draw bounding box
    box = bounding_box.as_int()
    cv.drawContours(image, [box], 0, (0,255,0), thickness=2)

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

    if side == 0: # laplacian_side
        cv.polylines(image, [half_1], True, color=(0,0,255), thickness=2)
    else:
        cv.polylines(image, [half_2], True, color=(0,0,255), thickness=2)

    return image
