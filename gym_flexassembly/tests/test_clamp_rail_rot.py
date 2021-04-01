#!/usr/bin/python

"""This is a test for the snapping of a clamp on the rail.
:Author:
  `Dennis Leroy Wigand <dwigand@cor-lab.de>`
"""

import pybullet as p
import time
import numpy as np
import math

# FLEX ASSEMBLY DATA IMPORTS
from gym_flexassembly import data as flexassembly_data
# print("flexassembly_data.getDataPath() = " + str(flexassembly_data.getDataPath()))

client = p.connect(p.GUI_SERVER)
p.setGravity(0, 0, -9.81, physicsClientId=client)
# p.setGravity(0, 0, 0, physicsClientId=client)

p.setAdditionalSearchPath(flexassembly_data.getDataPath())

planeId = p.loadURDF("objects/plane_solid.urdf")

rail = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/rail_scale.urdf", useFixedBase=True)
p.resetBasePositionAndOrientation(rail, [-0.5, 0, 1], [0,0,0,1])

c = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/prim_clamp_test.urdf", useFixedBase=False)
p.resetBasePositionAndOrientation(c, [0, -0.13, 1.15], [0.707,0,0,0.707])



# rail_small = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/rail.urdf", useFixedBase=True, globalScaling=1)
# p.resetBasePositionAndOrientation(rail_small, [-0.2, -0, 0.5], [0.707,0,0,0.707])

# c_test = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/smartobjects/W_QS_1/W_QS_1_test.urdf", useFixedBase=False, globalScaling=1)
# p.resetBasePositionAndOrientation(c_test, [0, -0.0208, 0.5166], [0.707,0,0,0.707])

rail_small = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/rail.urdf", useFixedBase=True, globalScaling=10)
p.resetBasePositionAndOrientation(rail_small, [-0.2, -3, 0.5], [0.707,0,0,0.707])

c_test = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/smartobjects/W_QS_1/W_QS_1_test.urdf", useFixedBase=False, globalScaling=10)
p.resetBasePositionAndOrientation(c_test, [0, -3.21, 0.664], [0.707,0,0,0.707])

rail_small = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/rail.urdf", useFixedBase=True, globalScaling=1)
p.resetBasePositionAndOrientation(rail_small, [-0.2, -0, 0.5], [0,0,0,1])

c_test = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/smartobjects/W_QS_1/W_QS_1_test.urdf", useFixedBase=False, globalScaling=1)
p.resetBasePositionAndOrientation(c_test, [0, 0.02, 0.523], [0,0,0,1])

# ball = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/flexassembly/ball.urdf", useFixedBase=False)
# p.resetBasePositionAndOrientation(ball, [0.05, 0.005, 0.04], [0,0,0,1])

p.setTimeStep(0.001) # TODO DLW

while 1:
    p.stepSimulation()
    # time.sleep(0.05)


