#! /usr/bin/env python
from __future__ import division

import sys
import copy
import time

import math
import numpy as np
import cv2

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

import rospy

import tf
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool, Float64MultiArray

from trac_ik_python.trac_ik import IK

#######################
#######################

class TracIKPublisher():

    def __init__(self):

        ##################################
        # TF transforms
        ##################################

        # Transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transform_broadcaster = tf2_ros.TransformBroadcaster()

        # WAM links / joints
        self.base_link_frame = 'wam/base_link'
        self.forearm_link_frame = 'wam/forearm_link'
        self.ee_link_frame = 'wam/wrist_palm_stump_link'  # 'wam/bhand/bhand_grasp_link' or 'wam/wrist_palm_link'
        self.zed_link_frame = 'zed_camera_center'   # 'zed_camera_center' or 'camera_frame'
        self.camera_link_frame = 'camera_frame'
        self.object_frame = 'object_frame'
        self.aruco_frame = 'aruco_frame'

        #######################
        # DEMO Sub-Tasks
        #######################

        self.demo_sub_tasks_summit_sub = rospy.Subscriber("/task_completion_flag_summit", Bool, self.demo_sub_tasks_callback)

        self.demo_sub_tasks_summit_pub = rospy.Publisher("/task_completion_flag_wam", Bool, queue_size=1)
        self.demo_sub_tasks_wam_pub = rospy.Publisher("/task_completion_flag_wam", Bool, queue_size=1)

        self.is_summit_in_position = False

        #######################
        #######################

        # Trac IK
        self.ik_solver = IK(self.base_link_frame, self.ee_link_frame)
        # self.seed_state = [0.00] * self.ik_solver.number_of_joints
        self.seed_state = [0.00010957005627754578, 0.8500968575130496, -0.00031928475261511213, 1.868559041954476, 0.0, -0.0006325693970662439, -0.00030823458564346445]

        #######################
        # Command Barret Arm
        #######################

        self.command_barrett_arm_pub = rospy.Publisher("/arm_position_controller/command", Float64MultiArray, queue_size=1)

        self.is_barrett_capture = False
        self.capture_joint_positions = [0.00695301832677738,
                                     -0.4587789565136406,
                                     -0.002222416045176924,
                                     2.208318148967319,
                                     0.027892071199038658,
                                     -0.1788168083040264,
                                     -0.028431769350072793]

        self.home_joint_positions = [0, -1.25, 0, 3, 0.785, 0, 0]

        # self.is_barrett_home = False
        # self.send_barrett_to_joint_positions(self.home_joint_positions, barrett_arm_state='Home')

        #######################
        # Object Pose
        #######################

        # self.object_pose_sub = rospy.Subscriber("/aruco_single/pose", PoseStamped, self.object_pose_callback)

        #######################
        # Trac IK
        #######################

        while True:

            position = np.array([0.5, 0, 0.85])
            orientation = np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)])  # perpendicular to the floor

            wam_joint_states = self.ik_solver.get_ik(self.seed_state,
                                                     #########################
                                                     position[0],
                                                     position[1],
                                                     position[2],
                                                     orientation[0],
                                                     orientation[1],
                                                     orientation[2],
                                                     orientation[3],
                                                     #########################
                                                     # brx=0.5, bry=0.5, brz=0.5
                                                     )

            wam_joint_states = np.array(wam_joint_states, dtype=float).reshape(-1).tolist()
            print("[trac-ik]   /wam/joint_states: ", wam_joint_states)
            # if len(wam_joint_states) == 1:
            #     rospy.logwarn("IK solver failed ..")
            #     return

            self.send_barrett_to_joint_positions(joint_positions=wam_joint_states,
                                                 barrett_arm_state="before arUco")
            # time.sleep(5)

            # home
            # self.is_barrett_home = False
            # self.send_barrett_to_joint_positions(self.home_joint_positions, barrett_arm_state='Home')

    #######################
    #######################

    def send_barrett_to_joint_positions(self, joint_positions, barrett_arm_state='Home'):

        rospy.loginfo("")
        rospy.loginfo("moving to --> {} ..".format(barrett_arm_state))

        for _ in range(10):
            msg = Float64MultiArray()
            msg.data = joint_positions
            self.command_barrett_arm_pub.publish(msg)
            time.sleep(0.1)

        if barrett_arm_state == 'Home' and joint_positions == self.home_joint_positions:
            time.sleep(5) # extra time to sleep if state is far from home state
            self.is_barrett_home = True
            self.is_barrett_capture = False

        elif barrett_arm_state == 'Capture' and joint_positions == self.capture_joint_positions:
            time.sleep(5)
            self.is_barrett_home = False
            self.is_barrett_capture = True

        rospy.loginfo("Barrett Arm is at --> {}!".format(barrett_arm_state))

    #######################
    #######################

    def demo_sub_tasks_callback(self, is_summit_in_position_msg):

        if is_summit_in_position_msg.data:
            self.is_summit_in_position = True
            rospy.loginfo("")
            rospy.loginfo("Summit is in position for Barrett Arm operations!")
            self.send_barrett_to_joint_positions(self.capture_joint_positions, barrett_arm_state='Capture')

    #######################
    #######################

    def object_pose_callback(self, object_in_world_frame_msg):

        #######################
        #######################

        if self.is_barrett_home and not self.is_barrett_capture:
            return

        #######################
        #######################

        x = object_in_world_frame_msg.pose.position.x
        x -= 15/100 # offset for: 'wam/wrist_palm_link'
        y = object_in_world_frame_msg.pose.position.y
        z = object_in_world_frame_msg.pose.position.z
        position = np.array([x, y, z])

        w = object_in_world_frame_msg.pose.orientation.w
        x = object_in_world_frame_msg.pose.orientation.x
        y = object_in_world_frame_msg.pose.orientation.y
        z = object_in_world_frame_msg.pose.orientation.z
        orientation = np.array([x, y, z, w])

        #######################
        # modify marker value in sim
        #######################

        position = np.array([0.5, 0, 0.85])
        orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]) # perpendicular to the floor

        self.print_object_pose(t=position, q=orientation)

        #######################
        # pub tf
        #######################

        # tf
        aruco_frame_on_object = geometry_msgs.msg.TransformStamped()
        aruco_frame_on_object.header.frame_id = self.base_link_frame
        aruco_frame_on_object.child_frame_id = self.aruco_frame
        aruco_frame_on_object.header.stamp = rospy.Time.now()
        aruco_frame_on_object.transform.translation.x = position[0]
        aruco_frame_on_object.transform.translation.y = position[1]
        aruco_frame_on_object.transform.translation.z = position[2]
        aruco_frame_on_object.transform.rotation.x = orientation[0]
        aruco_frame_on_object.transform.rotation.y = orientation[1]
        aruco_frame_on_object.transform.rotation.z = orientation[2]
        aruco_frame_on_object.transform.rotation.w = orientation[3]
        self.transform_broadcaster.sendTransform(aruco_frame_on_object)

        #######################
        # 1. capture --> before aruco
        #######################

        before_aruco_position = position.copy()
        before_aruco_position[0] -= 15 / 100                                    # offset for: 'wam/wrist_palm_link'
        before_aruco_position[0] = before_aruco_position[0] - 10/100            # before aruco
        before_aruco_orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]) # perpendicular to the floor

        wam_joint_states = self.ik_solver.get_ik(self.seed_state,
                                                #########################
                                                before_aruco_position[0],
                                                before_aruco_position[1],
                                                before_aruco_position[2],
                                                before_aruco_orientation[0],
                                                before_aruco_orientation[1],
                                                before_aruco_orientation[2],
                                                before_aruco_orientation[3],
                                                #########################
                                                # brx=0.5, bry=0.5, brz=0.5
                                                )

        wam_joint_states = np.array(wam_joint_states, dtype=float).reshape(-1).tolist()
        # print("[trac-ik]   /wam/joint_states: ", wam_joint_states)
        if len(wam_joint_states) == 1:
            rospy.logwarn("IK solver failed ..")
            return

        self.send_barrett_to_joint_positions(joint_positions=wam_joint_states,
                                             barrett_arm_state="before arUco")
        time.sleep(5)

        #######################
        # pub tf
        #######################

        # tf
        aruco_frame_on_object = geometry_msgs.msg.TransformStamped()
        aruco_frame_on_object.header.frame_id = self.base_link_frame
        aruco_frame_on_object.child_frame_id = self.aruco_frame
        aruco_frame_on_object.header.stamp = rospy.Time.now()
        aruco_frame_on_object.transform.translation.x = position[0]
        aruco_frame_on_object.transform.translation.y = position[1]
        aruco_frame_on_object.transform.translation.z = position[2]
        aruco_frame_on_object.transform.rotation.x = orientation[0]
        aruco_frame_on_object.transform.rotation.y = orientation[1]
        aruco_frame_on_object.transform.rotation.z = orientation[2]
        aruco_frame_on_object.transform.rotation.w = orientation[3]
        self.transform_broadcaster.sendTransform(aruco_frame_on_object)

        #######################
        # 2. before aruco -> at aruco
        #######################

        at_aruco_position = position.copy()
        at_aruco_position[0] -= 15 / 100                                         # offset for: 'wam/wrist_palm_link'
        at_aruco_orientation = np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)])  # perpendicular to the floor

        wam_joint_states = self.ik_solver.get_ik(self.seed_state,
                                                #########################
                                                at_aruco_position[0],
                                                at_aruco_position[1],
                                                at_aruco_position[2],
                                                at_aruco_orientation[0],
                                                at_aruco_orientation[1],
                                                at_aruco_orientation[2],
                                                at_aruco_orientation[3],
                                                #########################
                                                # brx=0.5, bry=0.5, brz=0.5
                                                )

        wam_joint_states = np.array(wam_joint_states, dtype=float).reshape(-1).tolist()
        # print("[trac-ik]   /wam/joint_states: ", wam_joint_states)
        if len(wam_joint_states) == 1:
            rospy.logwarn("IK solver failed ..")
            return

        self.send_barrett_to_joint_positions(joint_positions=wam_joint_states,
                                             barrett_arm_state="at arUco")
        time.sleep(5)

        #######################
        # pub tf
        #######################

        # tf
        aruco_frame_on_object = geometry_msgs.msg.TransformStamped()
        aruco_frame_on_object.header.frame_id = self.base_link_frame
        aruco_frame_on_object.child_frame_id = self.aruco_frame
        aruco_frame_on_object.header.stamp = rospy.Time.now()
        aruco_frame_on_object.transform.translation.x = position[0]
        aruco_frame_on_object.transform.translation.y = position[1]
        aruco_frame_on_object.transform.translation.z = position[2]
        aruco_frame_on_object.transform.rotation.x = orientation[0]
        aruco_frame_on_object.transform.rotation.y = orientation[1]
        aruco_frame_on_object.transform.rotation.z = orientation[2]
        aruco_frame_on_object.transform.rotation.w = orientation[3]
        self.transform_broadcaster.sendTransform(aruco_frame_on_object)

        #######################
        # 3. home position
        #######################

        self.send_barrett_to_joint_positions(self.home_joint_positions, barrett_arm_state='Home')

        msg = Bool()
        msg.data = True
        self.demo_sub_tasks_wam_pub.publish(msg)

        msg = Bool()
        self.is_summit_in_position = False
        msg.data = self.is_summit_in_position
        self.demo_sub_tasks_summit_pub.publish(msg)

    #######################
    #######################


    def print_object_pose(self, t, q):

        # convert translation to [cm]
        t = t.copy() * 100

        rot = np.array(R.from_quat(q).as_dcm()).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(rot)
        rvec = rvec * 180 / np.pi
        rvec = np.squeeze(np.array(rvec)).reshape(-1)

        rospy.loginfo('')
        rospy.loginfo('Detected arUco marker:')
        rospy.loginfo('position    [cm]:  x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(t[0], t[1], t[2]))
        rospy.loginfo('orientation [deg]: x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(rvec[0], rvec[1], rvec[2]))

    #######################
    #######################

def main():

    rospy.init_node('barrett_trac_ik', anonymous=True)
    barrett_controller = TracIKPublisher()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main()