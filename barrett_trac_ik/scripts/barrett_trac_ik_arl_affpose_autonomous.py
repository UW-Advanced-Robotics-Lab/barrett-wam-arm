#! /usr/bin/env python
from __future__ import division

import sys
import copy

import numpy as np
import cv2

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

import time
import threading

import rospy

import tf
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg

from std_msgs.msg import Bool, Float64MultiArray, Int32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped

from trac_ik_python.trac_ik import IK

#######################
#######################

# adding path to define custom services.
ROOT_DIR = '/home/akeaveny/catkin_ws/src/BarretWamArm/barrett_trac_ik/'
sys.path.append(ROOT_DIR)

from barrett_trac_ik.srv import JointMove, PoseMove, OpenGrasp, CloseGrasp

#######################
#######################


class TracIKPublisher:

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
        self.gripper_link_frame = 'wam/wrist_palm_link'
        self.gripper_link_offset = 0.06 + 0.12  # [cm] offset from 'wam/wrist_palm_link' to 'wam/bhand/bhand_grasp_link'

        #######################
        #######################

        # Trac IK
        self.ik_solver = IK(self.base_link_frame, self.gripper_link_frame)
        # self.seed_state = [0.00] * self.ik_solver.number_of_joints
        self.seed_state = [0.000109, 0.850096, -0.000319, 1.868559, 0.0, -0.000632, -0.000308]

        #######################
        # Init Barrett Services.
        #######################

        # Init services
        self.barrett_srv_command_joints = rospy.ServiceProxy("/wam/joint_move", JointMove)
        self.barrett_srv_command_pose = rospy.ServiceProxy("/wam/pose_move", PoseMove)
        self.bhand_srv_close_grasp = rospy.ServiceProxy("/bhand/close_grasp", OpenGrasp)
        self.bhand_srv_open_grasp = rospy.ServiceProxy("/bhand/open_grasp", CloseGrasp)

        # Init default capture position.
        self.drop_joint_positions = [1.25, 0.5, 0, 1.85, 0, 0, 0] 
        self.capture_joint_positions = [0.0, -0.65, 0, 2.9, 0.985, 0.75, 0]
        # self.capture_joint_positions = [0, -0.15, 0, 2.8, 0.985, 0.75, 0]
        # self.capture_joint_positions = [0.0, -0.85, 0, 3.1, 1.2, 0, 0]

        self.send_gripper_to_capture_position(init_gripper=True)

        #######################
        # Grasp Strategies.
        #######################

        self.obj_ids_for_top_down_grasps = np.array([1, 2, 3, 4, 5, 10, 11])
        self.z_offset_for_top_down_grasp = 0.15
        self.z_offset_for_successful_grasp = 0.10

        self.obj_ids_for_forward_grasps = np.array([6, 7, 8, 9])
        self.x_offset_for_forward_grasp = 0.085  # X [cm] BEHIND object.

        #######################
        # Object Pose.
        #######################

        # Init default values.
        self.obj_id = 0
        self.obj_radius = 0

        self.pose_meas_idx = 0
        self.max_pose_meas_idx = 5

        #######################
        # Callbacks
        #######################

        # init subscriber for barret pose.
        self.barrett_pose = np.zeros(shape=7)
        self.barrett_pose_sub = rospy.Subscriber("/wam/pose", PoseStamped, self.get_barrett_pose_callback, queue_size=1)

        self.object_ids_sub = rospy.Subscriber('/arl_affpose_ros/object_ids_for_grasping', Int32MultiArray, self.get_obj_ids_for_callback, queue_size=1)

        self.is_barrett_grasping_pub = rospy.Publisher("~is_barrett_grasping_flag", Bool, queue_size=1)
        self.pub_is_barrett_grasping_flag(False)

        #######################
        # Finite State Manager.
        #######################

        # # Init state manager in a dedicated thread.
        # self.finite_state_manager = finite_state_manager.FSM(self.capture_joint_positions)
        # finite_state_manager_thread = threading.Thread(target=self.finite_state_manager.update_robot_state)
        # finite_state_manager_thread.daemon = True
        # finite_state_manager_thread.start()

        #######################
        #######################

        rospy.loginfo("")
        rospy.loginfo("Barrett arm is ready for grasping")
        rospy.loginfo("")

    #######################
    #######################

    def pub_is_barrett_grasping_flag(self, is_barrett_grasping):
        msg = Bool()
        msg.data = is_barrett_grasping
        for _ in range(25):
            self.is_barrett_grasping_pub.publish(msg)
            time.sleep(0.01)

    def get_barrett_pose_callback(self, barret_pose_msg):

        self.barrett_pose = np.array([barret_pose_msg.pose.position.x,
                                      barret_pose_msg.pose.position.y,
                                      barret_pose_msg.pose.position.z,
                                      barret_pose_msg.pose.orientation.x,
                                      barret_pose_msg.pose.orientation.y,
                                      barret_pose_msg.pose.orientation.z,
                                      barret_pose_msg.pose.orientation.w,
                                      ]).reshape(-1)

    def send_barrett_joint_positions_cmd(self, joint_positions):

        msg = Float64MultiArray()
        msg.data = joint_positions
        self.barrett_srv_command_joints.call(joint_positions)

    def send_barrett_pose_cmd(self, position, orientation):

        msg = Pose()
        msg.position.x = position[0]
        msg.position.y = position[1]
        msg.position.z = position[2]
        msg.orientation.x = orientation[0]
        msg.orientation.y = orientation[1]
        msg.orientation.z = orientation[2]
        msg.orientation.w = orientation[3]
        self.barrett_srv_command_pose.call(msg)

    #######################
    #######################

    def map_obj_id_to_name(self, object_id):

        if object_id == 1:  # 001_mallet
            return 'Mallet'
        elif object_id == 2:  # 002_spatula
            return 'Spatula'
        elif object_id == 3:  # 003_wooden_spoon
            return 'Wooden_spoon'
        elif object_id == 4:  # 004_screwdriver
            return 'Screwdriver'
        elif object_id == 5:  # 005_garden_shovel
            return 'Garden_shovel'
        elif object_id == 6:  # 019_pitcher_base
            return 'Pitcher'
        elif object_id == 7:  # 024_bowl
            return 'Bowl'
        elif object_id == 8:  # 025_mug
            return 'Mug'
        elif object_id == 9:  # 035_power_drill
            return 'Power_drill'
        elif object_id == 10:  # 037_scissors
            return 'Scissors'
        elif object_id == 11:  # 051_large_clamp
            return 'Large_clamp'
        else:
            print(" --- Object ID does not map to Object Label --- ")
            exit(1)

    def set_obj_radius(self, obj_id, obj_name):

        prev_obj_radius = self.obj_radius
        if obj_id in np.array([1, 2, 3, 4, 5, 10, 11]):
            self.obj_radius = 0.05  # [cm] for top down grasps.
        elif obj_id == 6:
            self.obj_radius = 0.1  # [cm]
        elif obj_id == 8:
            self.obj_radius = 0.035  # [cm]
        elif obj_id == 9:
            self.obj_radius = 0.035  # [cm]

        if prev_obj_radius != self.obj_radius:
            rospy.loginfo("-------> Setting object's radius to: {} [cm]".format(self.obj_radius))

    def set_obj_id(self, obj_id=0, obj_name='background'):

        prev_obj_id = self.obj_id
        self.obj_id = obj_id
        self.obj_name = obj_name
        self.grasp_frame = 'object_pose_for_grasping_{}'.format(self.obj_id)
        self.object_pose_sub = rospy.Subscriber(self.grasp_frame, PoseStamped, self.object_pose_callback, queue_size=1)

        if prev_obj_id != self.obj_id:
            print('')
            rospy.loginfo('-------> Detected: {}'.format(self.obj_name))
            rospy.loginfo('-------> subscribing to: /{}'.format(self.grasp_frame))

    def get_obj_ids_for_callback(self, msg):

        obj_ids = np.array(msg.data, dtype=int)
        # TODO: currently selecting the first object id from list.
        obj_id = obj_ids[0]
        obj_name = "{}".format(self.map_obj_id_to_name(obj_id))
        self.set_obj_id(obj_id, obj_name)
        self.set_obj_radius(obj_id, obj_name)

    #######################
    #######################

    def reset_barrett_arm_for_grasping(self):
        # Move the Barrett back to capture.
        self.send_gripper_to_capture_position()
        # Reset Pose Estimation Count.
        self.pose_meas_idx = 0

    def send_gripper_to_capture_position(self, init_gripper=False):

        if init_gripper:
            self.bhand_srv_close_grasp()
            time.sleep(1.25)

        # Send Barret to capture position.
        self.send_barrett_joint_positions_cmd(self.capture_joint_positions)
        time.sleep(3)
        # Ensure bhand is closed.
        self.bhand_srv_close_grasp()
        time.sleep(1.25)

    def send_gripper_to_drop_position(self):
        # Send Barret to capture position.
        self.send_barrett_joint_positions_cmd(self.drop_joint_positions)
        time.sleep(7)
        # Ensure bhand is closed.
        self.bhand_srv_open_grasp()
        time.sleep(1.25)

    def prepare_gripper_for_grasping(self, position, orientation):

        self.y_offset_for_aligning_orientation = 0.1  # [cm]

        # 1st) move arm up out of the way
        position_1 = self.barrett_pose[:3]
        orientation_1 = self.barrett_pose[3:]
        # move up in z direction by constant Z amount [cm]
        position_1[2] += self.z_offset_for_top_down_grasp  # [cm]

        # 2) set orientation.
        position_2 = position_1.copy()
        orientation_2 = orientation.copy()

        # send pose command.
        self.send_barrett_pose_cmd(position_1, orientation_1)
        time.sleep(2)

        # open hand
        self.bhand_srv_open_grasp()
        time.sleep(1)

        # get correct orientation.
        self.send_barrett_pose_cmd(position_2, orientation_2)
        time.sleep(5)

    #######################
    #######################

    def execute_top_down_grasp(self, position, orientation):

        #######################
        #######################

        # adding offset from 'wam/wrist_palm_link' to 'bhand/grasp_link'
        position[2] += self.gripper_link_offset

        # two stage approach:
        # 1. move gripper to a pose X [cm] above object.
        # 2. move gripper directly downwards to grasp object.
        position_1 = position.copy()
        position_1[2] += self.z_offset_for_top_down_grasp

        position_2 = position.copy()
        position_2[2] += self.obj_radius

        #######################
        #######################

        wam_joint_states_1 = self.ik_solver.get_ik(self.seed_state,
                                                   position_1[0], position_1[1], position_1[2],
                                                   orientation[0], orientation[1], orientation[2], orientation[3],
                                                   )

        wam_joint_states_2 = self.ik_solver.get_ik(self.seed_state,
                                                   position_2[0], position_2[1], position_2[2],
                                                   orientation[0], orientation[1], orientation[2], orientation[3],
                                                   )

        wam_joint_states_1 = np.array(wam_joint_states_1, dtype=float).reshape(-1)
        wam_joint_states_2 = np.array(wam_joint_states_2, dtype=float).reshape(-1)

        print('')
        if np.isnan(np.sum(wam_joint_states_1)):
            rospy.logwarn("IK solver failed on wam_joint_states_1 ..")
            self.reset_barrett_arm_for_grasping()
            return

        if np.isnan(np.sum(wam_joint_states_2)):
            rospy.logwarn("IK solver failed on wam_joint_states_2 ..")
            self.reset_barrett_arm_for_grasping()
            return

        rospy.loginfo("Found valid solution for IK")

        #######################
        #######################

        # TODO: Determine when we have a reliable pose to grasp (i.e. Kalman Filter).
        self.prepare_gripper_for_grasping(position, orientation)

        # Move the barrett above the object.
        self.send_barrett_joint_positions_cmd(wam_joint_states_1)
        time.sleep(10)

        # Move the barrett down to the object.
        self.send_barrett_joint_positions_cmd(wam_joint_states_2)
        time.sleep(5)

        # grasp object.
        self.bhand_srv_close_grasp()
        time.sleep(1)

        #######################
        #######################

        # test successful grasp.
        position_1 = self.barrett_pose[:3]
        orientation_1 = self.barrett_pose[3:]
        # move up in z direction
        position_1[2] += self.z_offset_for_successful_grasp
        self.send_barrett_pose_cmd(position_1, orientation_1)
        time.sleep(3)

        # move arm up out of the way
        position_1 = self.barrett_pose[:3]
        orientation_1 = self.barrett_pose[3:]
        # move up in z direction
        position_1[2] -= self.z_offset_for_successful_grasp
        self.send_barrett_pose_cmd(position_1, orientation_1)
        time.sleep(3)

        # grasp object.
        self.bhand_srv_open_grasp()
        time.sleep(1)

        # drop object
        # self.send_gripper_to_drop_position()

        #######################
        #######################

        # move arm up out of the way
        position_1 = self.barrett_pose[:3]
        orientation_1 = self.barrett_pose[3:]
        # move up in z direction
        position_1[2] += self.z_offset_for_successful_grasp
        self.send_barrett_pose_cmd(position_1, orientation_1)
        time.sleep(3)

        self.reset_barrett_arm_for_grasping()
        time.sleep(3)
        rospy.loginfo("Completed Grasp!")

    def execute_forward_grasp(self, position, orientation):

        #######################
        #######################

        # three stage approach:
        # 1. move gripper to a pose X [cm] above object.
        # 2. move gripper directly downwards to grasp object.
        # 3. move gripper directly forwards to grasp object.
        position_1 = position.copy()
        # adding offset from 'wam/wrist_palm_link' to 'bhand/grasp_link'
        position_1[0] -= (self.gripper_link_offset + self.x_offset_for_forward_grasp + self.obj_radius)
        position_1[2] += self.z_offset_for_top_down_grasp

        position_2 = position_1.copy()
        position_2[2] -= self.z_offset_for_top_down_grasp

        position_3 = position_2.copy()
        position_3[0] += self.obj_radius

        #######################
        #######################

        wam_joint_states_1 = self.ik_solver.get_ik(self.seed_state,
                                                   position_1[0], position_1[1], position_1[2],
                                                   orientation[0], orientation[1], orientation[2], orientation[3],
                                                   )

        wam_joint_states_2 = self.ik_solver.get_ik(self.seed_state,
                                                   position_2[0], position_2[1], position_2[2],
                                                   orientation[0], orientation[1], orientation[2], orientation[3],
                                                   )

        wam_joint_states_3 = self.ik_solver.get_ik(self.seed_state,
                                                   position_3[0], position_3[1], position_3[2],
                                                   orientation[0], orientation[1], orientation[2], orientation[3],
                                                   )

        wam_joint_states_1 = np.array(wam_joint_states_1, dtype=float).reshape(-1)
        wam_joint_states_2 = np.array(wam_joint_states_2, dtype=float).reshape(-1)
        wam_joint_states_3 = np.array(wam_joint_states_3, dtype=float).reshape(-1)

        if np.isnan(np.sum(wam_joint_states_1)):
            rospy.logwarn("IK solver failed on wam_joint_states_1 ..")
            self.reset_barrett_arm_for_grasping()
            return

        # print('')
        if np.isnan(np.sum(wam_joint_states_2)):
            rospy.logwarn("IK solver failed on wam_joint_states_2 ..")
            self.reset_barrett_arm_for_grasping()
            return

        if np.isnan(np.sum(wam_joint_states_3)):
            rospy.logwarn("IK solver failed on wam_joint_states_3 ..")
            self.reset_barrett_arm_for_grasping()
            return

        rospy.loginfo("Found valid solution for IK")

        #######################
        #######################

        # TODO: Determine when we have a reliable pose to grasp (i.e. Kalman Filter).
        self.prepare_gripper_for_grasping(position, orientation)

        # Move the barret above the object for top down grasp.
        self.send_barrett_joint_positions_cmd(wam_joint_states_1)
        time.sleep(10)

        # Move the barrett down to the object.
        self.send_barrett_joint_positions_cmd(wam_joint_states_2)
        time.sleep(10)

        # Move the barrett to the object.
        self.send_barrett_joint_positions_cmd(wam_joint_states_3)
        time.sleep(5)

        # grasp object.
        self.bhand_srv_close_grasp()
        time.sleep(2)

        #######################
        #######################

        # move arm up out of the way
        position_1 = self.barrett_pose[:3]
        orientation_1 = self.barrett_pose[3:]
        # move up in z direction
        position_1[2] += self.z_offset_for_top_down_grasp
        self.send_barrett_pose_cmd(position_1, orientation_1)
        time.sleep(3)

        #######################
        #######################

        self.reset_barrett_arm_for_grasping()
        time.sleep(3)
        rospy.loginfo("Completed Grasp!")

    #######################
    #######################

    def object_pose_callback(self, object_pose_msg):

        if self.obj_id != 0 and self.pose_meas_idx < self.max_pose_meas_idx:

            #######################
            #######################

            self.pose_meas_idx += 1
            if self.pose_meas_idx == 1:
                print('')
            rospy.loginfo('-------> Detected: {}, {} time(s)'.format(self.obj_name, self.pose_meas_idx))

            #######################
            #######################

            x = object_pose_msg.pose.position.x
            y = object_pose_msg.pose.position.y
            z = object_pose_msg.pose.position.z
            position = np.array([x, y, z])

            x = object_pose_msg.pose.orientation.x
            y = object_pose_msg.pose.orientation.y
            z = object_pose_msg.pose.orientation.z
            w = object_pose_msg.pose.orientation.w
            orientation = np.array([x, y, z, w])

            #######################
            #######################
            if self.pose_meas_idx < self.max_pose_meas_idx:
                time.sleep(0.2)
            elif self.pose_meas_idx >= self.max_pose_meas_idx:

                #######################
                #######################

                # tell pose estimator to stop taking measurements.
                self.pub_is_barrett_grasping_flag(True)

                #######################
                #######################

                if self.obj_id in self.obj_ids_for_top_down_grasps:
                    self.execute_top_down_grasp(position, orientation)
                elif self.obj_id in self.obj_ids_for_forward_grasps:
                    self.execute_forward_grasp(position, orientation)

                #######################
                #######################

                # set obj id to background.
                self.set_obj_id()
                # tell pose estimator to start taking measurements.
                self.pub_is_barrett_grasping_flag(False)

def main():
    rospy.init_node('barrett_trac_ik', anonymous=True)
    ik = TracIKPublisher()

    # rate = rospy.Rate(50.0)
    # while not rospy.is_shutdown():
    #     rate.sleep()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        ik.pub_is_barrett_grasping_flag(False)
        print("Shutting down barrett_trac_ik node")


if __name__ == '__main__':
    main()