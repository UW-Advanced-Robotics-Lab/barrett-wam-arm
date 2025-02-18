#!/usr/bin/env python3

import rospy
from barrett_wam_msgs.msg import RTJointVel

def publish_joint_vel():
    # Initialize the node
    rospy.init_node('jnt_vel_publisher', anonymous=True)
    
    # Create a publisher for the /wam/jnt_vel_cmd topic
    pub = rospy.Publisher('/wam/jnt_vel_cmd', RTJointVel, queue_size=10)
    
    # Set the publish frequency (e.g., 10 Hz)
    rate = rospy.Rate(10)
    
    # Create the message
    jnt_vel_msg = RTJointVel()
    jnt_vel_msg.velocities = [0.5, 0.0, 0.0, 0.0]  # Set joint velocities to 0 (no movement)
    
    # Record the start time
    start_time = rospy.get_time()
    
    # Keep publishing for 1 second
    while not rospy.is_shutdown():
        # Publish the message continuously
        pub.publish(jnt_vel_msg)
        
        # Check if 1 second has passed
        if rospy.get_time() - start_time > 1.0:
            break
        
        # Sleep to maintain the publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_joint_vel()
    except rospy.ROSInterruptException:
        pass
