import io

import cv2 as cv
import numpy as np


def figure_to_img(fig, dpi=100):
    """
    Convert a pyplot figure to a numpy array/opencv image.
    """
    with io.BytesIO() as io_buf:
        fig.savefig(io_buf, format='raw', dpi=dpi)
        io_buf.seek(0)
        img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        return cv.cvtColor(img[:, :, :3], cv.COLOR_RGB2BGR)


def display_depth_image(img_depth):
    """
    Compute a colored representation of a depth image
    """
    img_depth = cv.convertScaleAbs(img_depth, alpha=0.3)
    img_depth = cv.equalizeHist(img_depth)
    img_depth = cv.applyColorMap(img_depth, cv.COLORMAP_JET)
    return img_depth


def visualize_plane(mask, img_depth, depth_plane, ax, interval=[0.25, 0.75], c='r'):
    """
    Visualize a depth plane computed from the depth image img_depth with
    the given mask.
    """
    depths = img_depth[mask]
    args = np.argsort(depths)
    args = args[int(interval[0] * args.shape[0]):int(interval[1] * args.shape[0])]
    xs, ys = np.nonzero(mask)

    _x = np.arange(xs.min(), xs.max(), (xs.max() - xs.min()) / 11)
    _y = np.arange(ys.min(), ys.max(), (ys.max() - ys.min()) / 11)
    xx, yy = np.meshgrid(_x, _y)

    ax.scatter(xs[args], ys[args], depths[args], c=c, s=50)
    inputs = np.array([xx.ravel(),yy.ravel()])
    outputs = depth_plane.predict(inputs.T).reshape(xx.shape)
    ax.plot_surface(xx, yy, outputs, rstride=1, cstride=1, alpha=0.2)


def visualize_pose(pos, orn, intrin, img):
    """
    Visualize a pose as a coordinate system by projection a
    unit coordinate system into the given image.
    """
    tvec = pos
    rvec, _ = cv.Rodrigues(orn.as_matrix())
    cam_matrix = np.array([[intrin.fx,         0, intrin.ppx],
                           [        0, intrin.fy, intrin.ppy],
                           [        0,         0,          1]])
    points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]) / 25
    points, _ = cv.projectPoints(points, rvec, tvec, cam_matrix, distCoeffs=None)
    points = np.intp(points[:, 0, :])

    cv.circle(img, tuple(points[-1]), 2, (0, 0, 0), -1)
    cv.line(img, tuple(points[-1]), tuple(points[0]), (0, 0, 255), 3)
    cv.line(img, tuple(points[-1]), tuple(points[1]), (0, 255, 0), 2)
    cv.line(img, tuple(points[-1]), tuple(points[2]), (255, 0, 0), 2)
    return img
