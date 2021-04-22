#!/usr/bin/env python
# license removed for brevity
import rospy
from geometry_msgs.msg import Pose

def talker():
    pub = rospy.Publisher('chatter', Pose, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(100) # 10hz
    while not rospy.is_shutdown():
        p = Pose()
        p.position.x = 10
        pub.publish(p)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass