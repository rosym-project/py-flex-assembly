import rospy

from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

import serial

class GripperServer(object):
    def __init__(self, name):
        print("Make sure to have access to device:\n\te.g., sudo chmod 777 /dev/ttyUSB0\n")
        #self.ser.open()
        self.ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=2,)
        self.ser.write([int('00000011', 2)])
        self.ser.close()
        rospy.init_node(name)
        self.a = rospy.Service(str(name) + '/open_gripper', Empty, self.open_gripper)
        self.b = rospy.Service(str(name) + '/close_gripper', Empty, self.close_gripper)
        print("Providing ROS service open_gripper at /" + str(name) + "/open_gripper using std_srvs.Empty")
        print("Providing ROS service close_gripper at /" + str(name) + "/close_gripper using std_srvs.Empty")
        print("")
        rospy.spin()

    def open_gripper(self, req):
        self.ser.open()
        self.ser.write([int('00000011', 2)])
        rospy.sleep(0.01)
        self.ser.write([int('00000001', 2)])
        self.ser.close()
        print("Gripper Opened")
        return []

    def close_gripper(self, req):
        self.ser.open()
        self.ser.write([int('00000011', 2)])
        rospy.sleep(0.01)
        self.ser.write([int('00000010', 2)])
        self.ser.close()
        print("Gripper Closed")
        return []

if __name__ == "__main__":
    gs = GripperServer("gripper1")