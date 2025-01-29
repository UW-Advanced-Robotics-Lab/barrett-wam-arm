#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped

def publish_goal():
    # Initialize the ROS node
    rospy.init_node('goal_publisher', anonymous=True)
    pub = rospy.Publisher('/uwarl/move_base_simple/goal', PoseStamped, queue_size=10)
    
    # Wait for publisher connection
    rospy.sleep(1)

    # Create PoseStamped message
    goal_msg = PoseStamped()
    goal_msg.header.frame_id = "uwarl_map"
    goal_msg.header.stamp = rospy.Time.now()
    
    # Set position
    goal_msg.pose.position.x = 1.279
    goal_msg.pose.position.y = -1.354
    goal_msg.pose.position.z = 0.0  # Assuming 2D movement
    
    # Set orientation (quaternion)
    goal_msg.pose.orientation.x = 0.0
    goal_msg.pose.orientation.y = 0.0
    goal_msg.pose.orientation.z = 0.361
    goal_msg.pose.orientation.w = 0.9325
    
    # Publish the message
    rospy.loginfo("Publishing goal: x=1.279, y=-1.354, quat_z=0.361, quat_w=0.9325")
    pub.publish(goal_msg)

if __name__ == "__main__":
    try:
        publish_goal()
    except rospy.ROSInterruptException:
        pass
