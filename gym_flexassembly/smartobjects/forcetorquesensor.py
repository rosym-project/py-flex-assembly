import os, inspect

import pybullet as p
import numpy as np
import copy
import math

import sys
import threading

from time import perf_counter

class ForceTorqueSensor:
    def __init__(self, pybullet, name, model_id, link_id, use_real_interface=True):
        self._name = name
        self._model_id = model_id
        self._link_id = link_id
        self._use_real_interface = use_real_interface
        self._terminate_thread = False
        self._p = pybullet

        self._lock = threading.Lock()
        self._new_data_available = False
        self._thread = None

        self._ft_sensor_forces = [0,0,0,0,0,0]

        

        # self._start_time = perf_counter()
        # self._wait_frame_rate = 1.0 / self._settings['framerate']

        self._p.enableJointForceTorqueSensor(self._model_id, self._link_id, True)

        if self._use_real_interface:
            # load (optional) ROS imports if the real interface should be mirrored
            try:
                import rospy
                from geometry_msgs.msg import Wrench
                from geometry_msgs.msg import Vector3

                # not sure if we really need this
                global rospy, Wrench, Vector3

                # create ros publisher
                self._pub_ft = rospy.Publisher('~ft/' + self._name, Wrench, queue_size=1)
                
                self._rate = rospy.Rate(500)

                print("\n\t> Initialized ft " + str(self._name) + " \n\t(geometry_msgs.Wrench) on ~ft/"+str(self._name)+" publisher\n")

                def thread_func():
                    while not self._terminate_thread:

                        # if self._new_data_available:

                        self._lock.acquire()
                        out_ft_wrench = Wrench()
                        out_ft_wrench.force.x = self._ft_sensor_forces[0]
                        out_ft_wrench.force.y = self._ft_sensor_forces[1]
                        out_ft_wrench.force.z = self._ft_sensor_forces[2]
                        out_ft_wrench.torque.x = self._ft_sensor_forces[3]
                        out_ft_wrench.torque.y = self._ft_sensor_forces[4]
                        out_ft_wrench.torque.z = self._ft_sensor_forces[5]
                        self._lock.release()

                        # publish the images
                        self._pub_ft.publish(out_ft_wrench)

                            # self._new_data_available = False

                        self._rate.sleep()

                # Start the ROS publishing thread
                self._thread = threading.Thread(target=thread_func, daemon=True)
                self._thread.start()
            
            except ImportError:
                print("ERROR IMPORTING ros camera", file=sys.stderr)
        
        self.reset()

    def update(self):
        if not self._terminate_thread:
            # if (perf_counter() - self._start_time) < self._wait_frame_rate:
            #     return

            # self._start_time = perf_counter()

            _, _, ft_sensor_forces, _ = self._p.getJointState(self._model_id, self._link_id)

            self._lock.acquire()
            self._ft_sensor_forces = ft_sensor_forces
            self._lock.release()

            # self._new_data_available = True

    def getModelId(self):
        return self._model_id

    def reset(self):
        pass

    def getUUid(self):
        return self._model_id

    def getObservation(self):
        observation = []
        return observation

    def terminate(self):
        self._terminate_thread = True
        self._thread.join()

        self._pub_ft.unregister()

    def __del__(self):
        self.terminate()

__all__ = ['ForceTorqueSensor']