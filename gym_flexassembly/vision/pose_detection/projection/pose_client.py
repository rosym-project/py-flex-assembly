import matplotlib.pyplot as plt
import numpy as np
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation as R

import rospy
from cosima_msgs.srv import GetClamp, GetClampResponse

tm = TransformManager()

rospy.wait_for_service('pose_estimation')
try:
    estimate = rospy.ServiceProxy('pose_estimation', GetClamp)
    estimation = estimate().i_pose
    pos = estimation.position
    pos = np.array([pos.x, pos.y, pos.z])
    _pos = pos * 1000
    print(f'Pos [{_pos[0]:.2f}, {_pos[1]:.2f}, {_pos[2]:.2f}]')

    orn = estimation.orientation
    orn = [orn.x, orn.y, orn.z, orn.w]
    print(f'Orn: {orn}')
    orn = R.from_quat(orn)
    _orn = orn.as_euler('zyx', degrees=True)
    print(f'Orn [{_orn[0]:.2f}, {_orn[2]:.2f}, {_orn[2]:.2f}]')

    orn = estimation.orientation
    orn = [orn.w, orn.x, orn.y, orn.z]
    transform = pt.transform_from_pq(np.hstack((pos, orn)))
    tm.add_transform('clamp', 'world', transform)
    ax = tm.plot_frames_in('world', s=0.2)
    ax.set_xlim((-0.75, 0.05))
    ax.set_ylim((-0.75, 0.05))
    ax.set_zlim((0.0, 1.0))
    plt.show()

except rospy.ServiceException as e:
    print(f'Service call failed... {repr(r)}')
