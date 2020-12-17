#!/usr/bin/python

"""This is a test for spawning the IIT COMAN robot.
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

coman_robot = p.loadURDF("/home/flex/system/flexassembly_dev_ws/src/py-flex-assembly/gym_flexassembly/data/robots/coman/urdf/coman.urdf", useFixedBase=False)
p.resetBasePositionAndOrientation(coman_robot, [0, 0, 0.6], [0,0,0,1])


p.setTimeStep(0.001) # TODO DLW

while 1:
    p.stepSimulation()
