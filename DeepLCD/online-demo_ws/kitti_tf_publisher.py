#!/usr/bin/env python
# license removed for brevity
import rospy
import tf
from std_msgs.msg import String, Header, ColorRGBA
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Point, Transform, Pose

import visualization_msgs.msg as vis_msg


def talker():
    rospy.init_node('kitti_tf_pub', anonymous=True)

    listener = tf.TransformListener()


    marker_pub_ = rospy.Publisher('viz_msgs_marker_publisher', vis_msg.Marker, latch=True, queue_size=10)

    pub = rospy.Publisher('kitti/transformStamped', TransformStamped, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():

        try:
            (trans,rot) = listener.lookupTransform('/world', '/camera_left', rospy.Time(0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        xform = TransformStamped()

        now = rospy.Time.now()
        xform.header = Header()

        xform.header.frame_id = '/world'
        xform.header.stamp = now
        xform.child_frame_id = '/camera_left'
        xform.transform.translation.x = trans[2]
        xform.transform.translation.y = trans[0]
        xform.transform.translation.z = trans[1]

        xform.transform.rotation.x = rot[0]
        xform.transform.rotation.y = rot[1]
        xform.transform.rotation.z = rot[2]
        xform.transform.rotation.w = rot[3]

        # hello_str = "hello world %s" % rospy.get_time()

        # rospy.loginfo(hello_str)
        
        pub.publish(xform)
        marker = vis_msg.Marker(type=vis_msg.Marker.SPHERE, ns='xyz', action=vis_msg.Marker.ADD, id = 0)
        marker.header.frame_id = '/world'
        marker.header.stamp = now
        marker.scale.x = 1000
        marker.scale.y = 1000
        marker.scale.z = 100
        marker.colors = [ColorRGBA(1.0 ,1.0 ,1.0 ,1.0)]
        # XYZ
        
        marker.pose.position = xform.transform.translation
        marker.pose.orientation = xform.transform.rotation
        marker.lifetime = rospy.Duration()
        marker_pub_.publish(marker)

        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
