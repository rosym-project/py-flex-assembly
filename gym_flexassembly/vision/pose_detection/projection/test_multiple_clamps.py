import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from gym_flexassembly.vision.pose_detection.projection.estimator import as_transform, FilterList
import gym_flexassembly.vision.pose_detection.projection.visualize as vis
import gym_flexassembly.vision.pose_detection.projection.features as fts

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

# create object to align depth and color images
config = rs.config()
config.enable_device_from_file('./realsense_recordings/multiple_clamps.bag')
pipeline = rs.pipeline()
pipeline.start(config)

tm = TransformManager()
pos_cam = np.array([-0.3, -0.3, 0.544 + 0.025])
orn_cam = R.from_euler('zyx', [90, 180, 0], degrees=True)
tm.add_transform("cam", "world", as_transform(pos_cam, orn_cam))

from sklearn import linear_model
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
    pos_cam = tm.get_transform('cam', 'world')[:3, -1]
    pos_table = pos_cam.copy()
    pos_table[2] = table_height + 0.01 # actually consider points slightly above the table
    orn_table = R.from_quat([0, 0, 0, 1])
    offsets = np.array([[0, 0.03, 0], [-0.03, -0.03, 0], [0.03, -0.03, 0]])

    pixels = []
    depths = []
    for i, offset in enumerate(offsets):
        coord_str = f'table_{i}'
        # add table point in world coordinates
        tm.add_transform(coord_str, 'world', as_transform(pos_table + offset, orn_table))
        # retrieve table point in cam coordinates
        pos_table_in_cam = tm.get_transform(coord_str, 'cam')[:3, -1]
        # compute pixel coordinates of point in depth image
        pixel = rs.rs2_project_point_to_pixel(intrin, pos_table_in_cam)
        pixels.append(pixel)
        # save depth value of table position
        depths.append(pos_table_in_cam[2])

        if i == 0:
            print(pos_table + offset)
            print(pos_table_in_cam)
            print(pixel)
    pixels = np.array(pixels)
    depths = np.array(depths)

    # fit plane to points
    inputs = np.array([pixels[:, 0], pixels[:,1]]).T
    depths = depths * 1000 # from meters to millimeters
    model = linear_model.LinearRegression().fit(inputs, depths)
    print(inputs)

    print(frame_depth.width)
    # compute plane depth image
    xx = np.arange(frame_depth.width)[::-1]
    yy = np.arange(frame_depth.height)[::-1]
    xx, yy = np.meshgrid(xx, yy)
    plane = model.intercept_ + model.coef_[1] * xx + model.coef_[0] * yy
    return plane


fl = FilterList()

cv.namedWindow('Display', cv.WINDOW_NORMAL)
cv.resizeWindow('Display', 1600, 900)

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
    return list(map(lambda cnt: fts.BoundingBox(cv.minAreaRect(cnt)), cnts))

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

depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
while True:
    frames = pipeline.wait_for_frames()
    frame_color = frames.get_color_frame()
    frame_depth = frames.get_depth_frame()
    frame_depth = fl.process(frame_depth).as_depth_frame()

    intrin = frame_depth.get_profile().as_video_stream_profile().get_intrinsics()

    # retrieve color and depth images
    # img_color = np.asanyarray(frame_color.get_data())
    # img_color = cv.cvtColor(img_color, cv.COLOR_RGB2BGR)
    # img_depth = np.asanyarray(frame_depth.get_data())

    # plane = compute_table_plane_image(tm, frame_depth)

    # # Only consider parts of the depth image over the table plane
    # img_depth = np.where(plane < img_depth, 0, img_depth)
    # img = img_depth / img_depth.max()
    # img = np.where(img > 0.1, 255, 0).astype(np.uint8)
    # # perform closing to fill small holes in depth img
    # img = cv.dilate(img, np.ones((3, 3)), iterations=2)
    # img = cv.erode(img, np.ones((3, 3)), iterations=2)
    # # find boxes
    # cnts, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # bbs = list(map(lambda cnt: fts.BoundingBox(cv.minAreaRect(cnt)), cnts))

    # # TODO: as another filter we could test, if the 3d bounding box matches in length/depth
    # def ratio_filter(bb):
        # try:
            # long_side_length = max(bb.w, bb.h)
            # short_side_length = min(bb.w, bb.h)
            # ratio = long_side_length / short_side_length
            # return 3 < ratio < 6
        # except ZeroDivisionError:
            # print('Ratio zero?', bb.w, bb.h)
            # print(bb)
            # return False

    # bbs = list(filter(ratio_filter, bbs))
    bbs = detect_bbs(tm, frame_depth)

    # retrieve color and depth images
    img_color = np.asanyarray(frame_color.get_data())
    img_color = cv.cvtColor(img_color, cv.COLOR_RGB2BGR)
    img_depth = np.asanyarray(frame_depth.get_data())

    scale_h = img_color.shape[1] / img_depth.shape[1]
    scale_w = img_color.shape[0] / img_depth.shape[0]

    img_depth = vis.display_depth_image(img_depth)

    for bb in bbs:
        bb_color = []
        cv.drawContours(img_depth, [bb.as_int()], -1, (0, 255, 0), 2)
        for _pt in bb.as_points():
            try:
                res = depth_to_color_pixel(_pt, frame_depth, frame_color)
                _res = np.array(res).astype(int)
                bb_color.append(_res)
                cv.circle(img_color, tuple(_res), 5, (0, 255, 0), -1)

                res = color_to_depth_pixel(res, frame_depth, frame_color, depth_scale)
                _res = np.array(res).astype(int)
                cv.circle(img_depth, tuple(_res), 3, (0, 0, 255), -1)
            except RuntimeError as e:
                print(e)
                print()
        bb_color = fts.BoundingBox(cv.minAreaRect(np.array(bb_color)))
        from gym_flexassembly.vision.pose_detection.projection.side_model import SidePredictor
        side = fts.detect_side_2(img_color, bb_color, SidePredictor('./models/side_model.pth'))

    img_depth = cv.resize(img_depth, img_color.shape[:2][::-1])
    cv.imshow('Display', np.vstack((img_color, img_depth)))


    print()
    # print(bb)
    # bb_refined = fts.refine_bb(bb, img_color)
    # cv.drawContours(img_color, [bb_refined.as_int()], -1, (0, 255, 0), 3)
    key = cv.waitKey(0)
    if key == ord('q'):
        print('Break..')
        exit(0)

