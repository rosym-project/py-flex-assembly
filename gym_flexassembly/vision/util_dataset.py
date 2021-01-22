import math
import random

import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

from gym_flexassembly.envs.flex_assembly_env import FlexAssemblyEnv


def setup_env():
    env = FlexAssemblyEnv(stepping=False, gui=True)
    env.remove_camera('global')
    for clamp_id in env.object_ids['clamps']:
        p.removeBody(clamp_id)
    for coordinate_id in env.object_ids['coordinate_systems']:
        p.removeBody(coordinate_id)

def get_image(camera_settings):
    _, _, rgba, _, _ = p.getCameraImage(
                              camera_settings['width'],
                              camera_settings['height'],
                              camera_settings['view_matrix'],
                              camera_settings['projection_matrix'],
                              shadow=True,
                              renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return rgba[:, :, :3]


def get_random_camera_view(target_pos, min_dist=0.2, max_dist=0.25, range_upper=0.05, range_lower=0.01):
    """
    Generate a random camera view depending on a target it looks on.

    Parameters
    ----------
    target_pos : [float]
        the position of the object looked at
    min_dist : float, optional
        the minimal distance of the camera position above (z-coordinate)
        the target object in meters (default: 0.2)
    max_dist : float, optional
        the maximal distance of the camera position above (z-coordinate)
        the target object in meters (default: 0.4)
    range_upper: float, optional
        range of how much the camera position in the x- and y-coordinate
        may differ from the target object's position (default: 0.3)
    range_lower: float, optional
        range of how much the x- and y-coordinate are changed from the target
        object's position for the look at vector of the camera (default: 0.1)
    """
    pos_x = target_pos[0] + random.uniform(-range_upper, range_upper)
    pos_y = target_pos[1] + random.uniform(-range_upper, range_upper)
    pos_z = target_pos[2] + random.uniform(min_dist, max_dist)
    pos = [pos_x, pos_y, pos_z]

    target_x = target_pos[0] + random.uniform(-range_lower, range_lower)
    target_y = target_pos[1] + random.uniform(-range_lower, range_lower)
    target_z = target_pos[2]
    target_pos = [target_x, target_y, target_z]

    up_x = random.uniform(-1, 1)
    up_y = random.uniform(-1, 1)
    up_z = random.uniform(-0.1, 0.1)
    up = [up_x, up_y, up_z]

    return {'pos': pos, 'target_pos': target_pos, 'up': up}


def get_random_camera_settings(target_pos, width=1280, height=720, fov=65, near=0.16, far=10, **kwargs):
    """
    Generate random camera settings by internally calling get_random_camera_view.
    """
    aspect = width / height
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    view = get_random_camera_view(target_pos, **kwargs)
    view_matrix = p.computeViewMatrix(view['pos'], view['target_pos'], view['up'])

    camera_settings = view
    camera_settings['view_matrix'] = view_matrix
    camera_settings['projection_matrix'] = projection_matrix
    camera_settings['width'] = width
    camera_settings['height'] = height
    camera_settings['aspect'] = aspect
    camera_settings['fov'] = fov
    camera_settings['near'] = near
    camera_settings['far'] = far
    return camera_settings


def generate_random_model_pose(table_offset_x=0.1, table_offset_y=1.2):
    """
    Generate a random model pose so that the model is placed on the table.
    """
    # orientation
    # always rotate around z
    # normal orn: [0, 0, x]
    # normal height: 0.71

    # low prob orn: [math.pi / 2, 0, x] or [3 * math.pi, 0, x]
    # height: 0.73

    # lying
    # orn [0, math.pi / 2, x] or [0, 3 * math.pi, x]
    # height: 0.705
    # base_orientation = random.randint(0, 2)
    # if base_orientation == 0:
        # # flat on the ground
        # if random.randint(0, 1) == 1:
            # roll = 0
            # z = 0.71
        # else:
            # roll = math.pi
            # z = 0.73
        # pitch = 0
        # z = 0.71
    # elif base_orientation == 1:
        # # long side on the ground
        # if random.randint(0, 1) == 0:
            # roll = math.pi / 2
            # z = 0.72
        # else:
            # roll = 3 * math.pi / 2
        # pitch = 0
        # z = 0.72
    # else:
        # # short side on the ground
        # roll = 0
        # if random.randint(0, 1) == 0:
            # pitch = math.pi / 2
        # else:
            # pitch = 3 * math.pi / 2
        # z = 0.73

    # roll = 0
    # if random.randint(0, 1) == 1:
        # roll = math.pi
    # pitch = 0
    # z = 0.71

    roll = 3 * math.pi / 2
    pitch = 0
    z = 0.72

    yaw = random.uniform(0, 2 * math.pi)
    rotation = p.getQuaternionFromEuler([roll, pitch, yaw])

    # table top goes from x[-1.07, -0.01] y[-0.03, -1.05]
    border_size = 0.25
    table_range_x = [-1.07 + border_size, -0.01 - border_size]
    table_range_y = [-1.05 + border_size, -0.03 - border_size]

    x = random.uniform(*table_range_x) + table_offset_x
    y = random.uniform(*table_range_y) + table_offset_y
    position = [x, y, z]

    return {'pos': position, 'orn': rotation}


def from_global_to_camera(pos, orn, view_matrix):
    _pos = view_matrix.dot(np.array([*pos, 1]))[:3]

    _orn = np.array(p.getMatrixFromQuaternion(orn)).reshape((3, 3))
    _orn = view_matrix[:3, :3].dot(_orn)
    _orn = R.from_matrix(_orn).as_quat()

    return _pos, _orn


def from_camera_to_global(pos, orn, view_matrix):
    return from_global_to_camera(pos, orn, np.linalg.inv(view_matrix))


if __name__ == '__main__':
    m = generate_random_model_pose()
    c = get_random_camera_settings(m['pos'])
    view_matrix = np.array(c['view_matrix']).reshape((4, 4)).transpose()

    _pos, _orn = from_global_to_camera(m['pos'], m['orn'], view_matrix)
    pos, orn = from_camera_to_global(_pos, _orn, view_matrix)

    print(f'Position diff {np.linalg.norm(pos - m["pos"]):.6f}')
    orn_matrx = np.array(p.getMatrixFromQuaternion(m['orn'])).reshape((3, 3))
    _orn_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape((3, 3))
    print(f'Rotation diff {np.linalg.norm(orn_matrx - _orn_matrix):.6f}')
