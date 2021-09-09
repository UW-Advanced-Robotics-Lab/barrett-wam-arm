#! /usr/bin/env python
from __future__ import division

import sys
import copy

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
from std_msgs.msg import Bool, Int32MultiArray

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

        ##################################
        # TF frames
        ##################################

        self.base_link_frame = 'wam/base_link'
        self.camera_link_frame = 'camera_frame'
        self.ee_link_frame = 'wam/wrist_palm_link'  # 'wam/wrist_palm_link' or 'wam/bhand/bhand_grasp_link'
        self.ee_link_offset = 0.06 + 0.12  # [cm] This is the offset to the from the ee frame to where the hand should close.
        self.offset_for_two_part_grasp_1 = 0.15  # [cm] This is an offset for a 'two-stage' / 'top-down' gripping approach.
        self.offset_for_two_part_grasp_2 = 0.025

        #######################
        #######################

        # Trac IK
        self.ik_solver = IK(self.base_link_frame, self.ee_link_frame)
        # self.seed_state = [0.00] * self.ik_solver.number_of_joints
        self.seed_state = [0.00010957005627754578, 0.8500968575130496, -0.00031928475261511213, 1.868559041954476, 0.0, -0.0006325693970662439, -0.00030823458564346445]

        #######################
        # Object Pose
        #######################

        # Init obj to background.
        self.obj_id = 0
        self.object_pose_sub = rospy.Subscriber('/arl_affpose_ros/object_ids_for_grasping', Int32MultiArray, self.get_obj_id_for_callback)

    #######################
    #######################

    def set_obj_id_for_callback(self, obj_id):
        """
         # 1:	001 - mallet
        # 2:	002_spatula
        # 3:	003_wooden_spoon
        # 4:	004_screwdriver
        # 5:	005_garden_shovel
        # 6:	019_pitcher_base
        # 7:	024_bowl
        # 8:	025_mug
        # 9:	035_power_drill
        # 10:	037_scissors
        # 11:	051_large_clamp
        """

        prev_obj_id = self.obj_id
        self.obj_id = obj_id
        self.grasp_frame = 'object_pose_for_grasping_{}'.format(self.obj_id)
        self.object_pose_sub = rospy.Subscriber(self.grasp_frame, PoseStamped, self.object_pose_callback)

        if prev_obj_id != self.obj_id:
            rospy.loginfo('-------> Setting Obj id to:{}'.format(self.obj_id))

    def get_obj_id_for_callback(self, msg):

        obj_ids = np.array(msg.data, dtype=int)
        # TODO: currently selecting the first object id from list.
        obj_id = obj_ids[0]
        self.set_obj_id_for_callback(obj_id)

    #######################
    #######################

    def object_pose_callback(self, object_pose_msg):

        #######################
        #######################

        x = object_pose_msg.pose.position.x
        y = object_pose_msg.pose.position.y
        z = object_pose_msg.pose.position.z + self.ee_link_offset
        position = np.array([x, y, z])

        w = object_pose_msg.pose.orientation.w
        x = object_pose_msg.pose.orientation.x
        y = object_pose_msg.pose.orientation.y
        z = object_pose_msg.pose.orientation.z
        orientation = np.array([x, y, z, w])

        #######################
        # Trac IK
        #######################

        wam_joint_states_1 = self.ik_solver.get_ik(self.seed_state,
                                                #########################
                                                position[0], position[1], position[2]+self.offset_for_two_part_grasp_1,
                                                orientation[0], orientation[1], orientation[2], orientation[3],
                                                #########################
                                                # brx=0.5, bry=0.5, brz=0.5
                                                )

        wam_joint_states_1 = np.array(wam_joint_states_1, dtype=float).reshape(-1)

        print("")
        if np.isnan(np.sum(wam_joint_states_1)):
            rospy.logwarn("IK solver failed ..")
            return

        wam_joint_states_2 = self.ik_solver.get_ik(self.seed_state,
                                                #########################
                                                position[0], position[1], position[2]+self.offset_for_two_part_grasp_2,
                                                orientation[0], orientation[1], orientation[2], orientation[3],
                                                #########################
                                                # brx=0.5, bry=0.5, brz=0.5
                                                )

        wam_joint_states_2 = np.array(wam_joint_states_2, dtype=float).reshape(-1)

        if np.isnan(np.sum(wam_joint_states_2)):
            rospy.logwarn("IK solver failed ..")
            return

        print("[trac-ik]   /wam/joint_states_1: ", wam_joint_states_1.tolist())
        print("[trac-ik]   /wam/joint_states_2: ", wam_joint_states_2.tolist())

def main():

    rospy.init_node('barrett_trac_ik', anonymous=True)
    TracIKPublisher()
    rate = rospy.Rate(5.0)
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main()