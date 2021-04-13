import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from pytransform3d import transformations as pt
from scipy.spatial.transform import Rotation as R
from sklearn import linear_model

def as_transform(pos, orn):
    """
    Convert pose into transform as used by pytransform3d
    """
    # get rotation as wxyz quaternion
    orn_q = orn.as_quat()
    orn_q = [orn_q[-1], *orn_q[:-1]]

    # create pose and return transform
    _pose = np.hstack((pos, orn_q))
    return pt.transform_from_pq(_pose)


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


def compute_table_plane_image(transform_manager, frame_depth, table_height=0.025):
    """
    Compute a depth image representing the table plane.
    This is done by entering three points of the table near to the camera position (in x,y)
    into the transform manager.
    Then, these points are retrieved in camera coordinates and projected onto the image
    retrieved by the camera.
    Afterwards a plane is fitted to these three points and an according depth image is
    created and returned.

    params:
    transform_manager: a transform manager containing a transformation from 'cam' to 'world'
    frame_depth: the depth frame as received from the realsense pipeline
    """
    intrinsics = frame_depth.profile.as_video_stream_profile().intrinsics
    pos_cam = transform_manager.get_transform('cam', 'world')[:3, -1]
    pos_table = pos_cam.copy()
    pos_table[2] = table_height + 0.01 # actually consider points slightly above the table
    orn_table = R.from_quat([0, 0, 0, 1])
    offsets = np.array([[0, 0.03, 0], [-0.03, -0.03, 0], [0.03, -0.03, 0]])

    pixels = []
    depths = []
    for i, offset in enumerate(offsets):
        coord_str = f'table_{i}'
        # add table point in world coordinates
        transform_manager.add_transform(coord_str, 'world', as_transform(pos_table + offset, orn_table))
        # retrieve table point in cam coordinates
        pos_table_in_cam = transform_manager.get_transform(coord_str, 'cam')[:3, -1]
        # compute pixel coordinates of point in depth image
        pixel = rs.rs2_project_point_to_pixel(intrinsics, pos_table_in_cam)
        pixels.append(pixel)
        # save depth value of table position
        depths.append(pos_table_in_cam[2])
    pixels = np.array(pixels)
    depths = np.array(depths)

    # fit plane to points
    inputs = np.array([pixels[:, 0], pixels[:,1]]).T
    depths = depths * 1000 # from meters to millimeters
    model = linear_model.LinearRegression().fit(inputs, depths)

    # compute plane depth image
    xx = np.arange(frame_depth.width)[::-1]
    yy = np.arange(frame_depth.height)[::-1]
    xx, yy = np.meshgrid(xx, yy)
    plane = model.intercept_ + model.coef_[1] * xx + model.coef_[0] * yy
    return plane


def compute_bbs(img_plane, frame_depth):
    """
    Compute bounding boxes of objects nearer to the camera than
    the plane image.
    """
    img_depth = np.asanyarray(frame_depth.get_data())
    img_depth = np.where(img_plane < img_depth, 0, img_depth)
    img = img_depth / img_depth.max()
    img = np.where(img > 0.1, 255, 0).astype(np.uint8)
    # perform closing to fill small holes in depth image
    img = cv.dilate(img, np.ones((3, 3)), iterations=2)
    img = cv.erode(img, np.ones((3, 3)), iterations=2)
    # find boxes
    cnts, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return list(map(lambda cnt: BoundingBox(cv.minAreaRect(cnt)), cnts))


def filter_bbs(bbs):
    """
    Filter bounding boxes by their ratio of long side to short side.
    The bounding box of a clamp should be inside 3 < ratio < 6.
    """
    # TODO: as another filter we could test, if the 3d bounding box matches in length/depth
    def ratio_filter(bb):
        try:
            long_side_length = max(bb.w, bb.h)
            short_side_length = min(bb.w, bb.h)
            ratio = long_side_length / short_side_length
            return 3 < ratio < 6
        except ZeroDivisionError:
            return False
    return list(filter(ratio_filter, bbs))


def detect_bbs(transform_manager, frame_depth):
    """
    Detect multiple bounding boxes around clamps in an image.
    """
    img_plane = compute_table_plane_image(transform_manager, frame_depth)
    bbs = compute_bbs(img_plane, frame_depth)
    return filter_bbs(bbs)


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

    return img_color[xs[0]:xs[1], ys[0]:ys[1]]

def detect_side_2(image, bb, side_model):
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

    img = img_from_half(image, half_1)
    pred = side_model.predict(img)
    
    # title = 'hole' if pred == 0 else 'other'
    # cv.imshow(title, img)

    return pred

def detect_side(image, box, visualize=True):
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

    if visualize:
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


def spiral(radius):
    """
    Create list to iterate in a spiral outwards.
    The values are sorted based on the distance to their center.
    E.g. [0, 1], [1, 0], [0, -1] and [-1, 0] will all appear before [1, 1].
    """
    N = 2 * radius - 1

    # Find the unique distances
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    G = np.sqrt(X**2 + Y**2)
    U = np.unique(G)

    # Identify these coordinates
    blocks = [[pair for pair in zip(*np.where(G == idx))] for idx in U if idx < N / 2]

    # Permute along the different orthogonal directions
    directions = np.array([[1,1], [-1,1], [1,-1], [-1,-1]])

    all_R = []
    for b in blocks:
        R = set()
        for item in b:
            for x in item * directions:
                R.add(tuple(x))
        R = np.array(list(R))

        # Sort by angle
        T = np.array([np.arctan2(*x) for x in R])
        R = R[np.argsort(T)]
        all_R.append(R)
    return np.concatenate(all_R) 


def depth_to_color_pixel(pixel, frame_depth, frame_color):
    """
    Map a pixel of a depth frame to a pixel in a non-aligned color frame.
    """
    intrin_depth = frame_depth.profile.as_video_stream_profile().intrinsics
    intrin_color = frame_color.profile.as_video_stream_profile().intrinsics

    extrin_depth_to_color = frame_depth.profile.get_extrinsics_to(frame_color.profile)
    extrin_color_to_depth = frame_color.profile.get_extrinsics_to(frame_depth.profile)

    # Note:
    # It can happen that the distance value at the exact pixel is invalid (== 0.0)
    # If this is the case, use a nearby valid pixel as an approximation
    _pixel = np.array(pixel)
    for s in spiral(5):
        depth = frame_depth.get_distance(*(_pixel + s).astype(int))
        if depth != 0.0:
            break

    if depth == 0.0:
        raise RuntimeError('Unkown depth value...')

    _pt = rs.rs2_deproject_pixel_to_point(intrin_depth, pixel, depth)
    _pt = rs.rs2_transform_point_to_point(extrin_depth_to_color, _pt)
    pixel_color = rs.rs2_project_point_to_pixel(intrin_color, _pt)
    return pixel_color

def color_to_depth_pixel(pixel, frame_depth, frame_color, depth_scale):
    """
    Map a pixel of a color frame to a pixel in a non-aligned depth frame.
    """
    intrin_depth = frame_depth.profile.as_video_stream_profile().intrinsics
    intrin_color = frame_color.profile.as_video_stream_profile().intrinsics

    extrin_depth_to_color = frame_depth.profile.get_extrinsics_to(frame_color.profile)
    extrin_color_to_depth = frame_color.profile.get_extrinsics_to(frame_depth.profile)

    pixel_depth = rs.rs2_project_color_pixel_to_depth_pixel(
            frame_depth.data,
            depth_scale,
            0.1,
            1.0,
            intrin_depth,
            intrin_color,
            extrin_depth_to_color,
            extrin_color_to_depth,
            pixel)
    return pixel_depth
