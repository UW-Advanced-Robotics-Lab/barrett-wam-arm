#! /usr/bin/env python
from __future__ import division

import os
import sys
import copy
import time

import math
import numpy as np
import cv2

from enum import IntEnum

from scipy.spatial.transform import Rotation as R

import rospy

import tf
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool, Float64MultiArray, Int8

ROOT_DIR = '/home/akeaveny/catkin_ws/src/barrett_trac_ik/'
sys.path.append(ROOT_DIR)
print("*** ROOT_DIR: {} ***".format(ROOT_DIR))

from barrett_trac_ik.srv import JointMove

from trac_ik_python.trac_ik import IK

#######################
#######################

class TracIKPublisher():
    class WARM_STATUS(IntEnum):
        # Will-publish:
        CORRIDOR_DOOR_BUTTON    = 1  # 1 : press the door button of the corridor
        ELEV_DOOR_BUTTON_CALL   = 2  # 2: press the elevator call button
        ELEV_DOOR_BUTTON_INSIDE = 3  # 3: press the floor button inside the elevator
        FAILED                  = -1 # -1: operation failed
        # Wont-publish:
        HOMING                  = 4  # 4: operation failed

    # Look up table for pre-calibrated joint-positions
    LUT_CAPTURED_JOINT_POSITIONS = {
        WARM_STATUS.ELEV_DOOR_BUTTON_INSIDE : [    
            0.00695301832677738,
            -0.4587789565136406,
            -0.002222416045176924,
            2.208318148967319,
            0.027892071199038658,
            -0.1788168083040264,
            -0.028431769350072793
        ],
        WARM_STATUS.ELEV_DOOR_BUTTON_CALL : [
            0.00695301832677738,
            -0.4587789565136406,
            -0.002222416045176924,
            2.208318148967319,
            0.027892071199038658,
            -0.1788168083040264,
            -0.028431769350072793
        ],
        WARM_STATUS.CORRIDOR_DOOR_BUTTON : [
            0.76,
            -0.4587789565136406,
            -0.002222416045176924,
            2.308318148967319,
            0.027892071199038658,
            -0.1788168083040264,
            -0.028431769350072793
        ],
        WARM_STATUS.FAILED : [0, -1.25, 0, 3, 0, 0, 0],
        WARM_STATUS.HOMING : [0, -1.25, 0, 3, 0, 0, 0]
    }


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

        self.demo_sub_tasks_summit_sub = rospy.Subscriber("/task_completion_flag_summit", Int8, self.demo_sub_tasks_callback)

        # self.demo_sub_tasks_summit_pub = rospy.Publisher("/task_completion_flag_wam", Int16, queue_size=1)
        self.demo_sub_tasks_wam_pub = rospy.Publisher("/task_completion_flag_wam", Int8, queue_size=1)

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

        # self.command_barrett_arm_pub = rospy.Publisher("/arm_position_controller/command", Float64MultiArray, queue_size=1)
        self.command_barrett_arm_srv = rospy.ServiceProxy("/wam/joint_move", JointMove)

        self.is_barrett_capture = False

        self.capture_joint_positions = self.LUT_CAPTURED_JOINT_POSITIONS[self.WARM_STATUS.HOMING]
        self.home_joint_positions = self.LUT_CAPTURED_JOINT_POSITIONS[self.WARM_STATUS.HOMING]

        self.is_barrett_home = False
        self.send_barrett_to_joint_positions(self.home_joint_positions, barrett_arm_state='Home')

        #######################
        # Object Pose
        #######################

        self.object_pose_sub = rospy.Subscriber("/aruco_single/pose", PoseStamped, self.object_pose_callback)

    #######################
    #######################

    def send_barrett_to_joint_positions(self, joint_positions, barrett_arm_state='Home'):

        rospy.loginfo("")
        rospy.loginfo("moving to --> {} ..".format(barrett_arm_state))

        for _ in range(10):
            msg = Float64MultiArray()
            msg.data = joint_positions
            # self.command_barrett_arm_pub.publish(msg)
            self.command_barrett_arm_srv.call(joint_positions)
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
        print("Demo sub tasks callback")
        if is_summit_in_position_msg.data:
            # interpret position request:
            self.warm_request = self.WARM_STATUS.FAILED
            try: 
                self.warm_request = self.WARM_STATUS(is_summit_in_position_msg.data)
            except ValueError:
                self.warm_request = self.WARM_STATUS.FAILED
            # update position request:
            self.capture_joint_positions = self.LUT_CAPTURED_JOINT_POSITIONS[warm_request]
            # perform:
            self.is_summit_in_position = True
            rospy.loginfo("")
            rospy.loginfo("Summit is in position for Barrett Arm operations! [REQUEST: {}]".format(warm_request))
            self.send_barrett_to_joint_positions(self.capture_joint_positions, barrett_arm_state='Capture')

    #######################
    #######################

    def object_pose_callback(self, object_in_world_frame_msg):

        #######################
        #######################

        if self.is_barrett_home and not self.is_barrett_capture:
            return

        time.sleep(5)

        #######################
        #######################

        x = object_in_world_frame_msg.pose.position.x  # /10
        y = object_in_world_frame_msg.pose.position.y  # /10
        z = object_in_world_frame_msg.pose.position.z  # /10
        # y += 13/100 # offset for: 'wam/wrist_palm_link'
        position = np.array([x, y, z])

        w = object_in_world_frame_msg.pose.orientation.w
        x = object_in_world_frame_msg.pose.orientation.x
        y = object_in_world_frame_msg.pose.orientation.y
        z = object_in_world_frame_msg.pose.orientation.z
        # orientation = np.array([w, x, y, z])
        orientation = np.array([x, y, z, w])

        # orientation = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0])

        self.print_object_pose(t=position, q=orientation)

        #######################
        # pub tf
        #######################
        # orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])

        # tf
        aruco_frame_on_object = geometry_msgs.msg.TransformStamped()
        aruco_frame_on_object.header.frame_id = self.camera_link_frame
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

        ##################################
        # Transforms
        ##################################
        time.sleep(3)
        try:

            # pose
            object_in_camera_frame_msg = PoseStamped()
            # object_in_camera_frame_msg.header.frame_id = self.camera_link_frame
            object_in_camera_frame_msg.pose.position.x = position[0]
            object_in_camera_frame_msg.pose.position.y = position[1]
            object_in_camera_frame_msg.pose.position.z = position[2]
            object_in_camera_frame_msg.pose.orientation.w = orientation[3]
            object_in_camera_frame_msg.pose.orientation.x = orientation[0]
            object_in_camera_frame_msg.pose.orientation.y = orientation[1]
            object_in_camera_frame_msg.pose.orientation.z = orientation[2]

            ''' object_T_world = object_T_zed * zed_T_world '''
            # zed_T_world
            camera_to_world = self.tf_buffer.lookup_transform(self.base_link_frame, self.camera_link_frame, rospy.Time(0))
            # object_T_world
            object_to_world = tf2_geometry_msgs.do_transform_pose(object_in_camera_frame_msg, camera_to_world)

            position = np.array([object_to_world.pose.position.x,
                                     object_to_world.pose.position.y,
                                     object_to_world.pose.position.z])

            orientation = np.array([
                                    object_to_world.pose.orientation.x,
                                    object_to_world.pose.orientation.y,
                                    object_to_world.pose.orientation.z,
                                    object_to_world.pose.orientation.w])

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Can't find transform from {} to {}".format(self.base_link_frame, self.camera_link_frame))
            return None

        # position[0] += 0.07
        # position[1] -= 0.16
        x_axis = np.array([1,0,0])
        y_axis = np.array([0,1,0])
        z_axis = np.array([0,0,1])
        # orientation = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0])
        # orientation = np.array([0,0,0,1])

        normal_dir = R.from_quat(orientation).apply(z_axis)
        tag_x_axis_dir = R.from_quat(orientation).apply(x_axis)
        tag_y_axis_dir = R.from_quat(orientation).apply(y_axis)
        # NOTE: Next two lines are for the elevator (outside) only (comment out for other locations)
        # position += -0.017 * tag_x_axis_dir
        # position += -0.177 * tag_y_axis_dir

        # NOTE: Next two lines are for the elevator (inside) only (comment out for other locations)
        position += 0.02 * tag_y_axis_dir
        position += -0.11 * tag_x_axis_dir

        print("Normal Dir: {}".format(normal_dir))
        # before_position = position.copy()
        # position[2] += 0.025
        before_position = position + 0.25 * normal_dir
        after_position = position + 0.125 * normal_dir
        print("Befpre position: {}, After position: {}".format(before_position, position))

        #######################
        # modify marker value in sim
        #######################

        # position = np.array([0.5, 0, 0.85])
        # orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]) # perpendicular to the floor

        # self.print_object_pose(t=position, q=orientation)

        #######################
        # pub tf
        #######################

        # # tf
        aruco_frame_on_object = geometry_msgs.msg.TransformStamped()
        aruco_frame_on_object.header.frame_id = self.base_link_frame
        aruco_frame_on_object.child_frame_id = 'aruco_frame_offset' # self.aruco_frame
        aruco_frame_on_object.header.stamp = rospy.Time.now()
        aruco_frame_on_object.transform.translation.x = before_position[0]
        aruco_frame_on_object.transform.translation.y = before_position[1]
        aruco_frame_on_object.transform.translation.z = before_position[2]
        aruco_frame_on_object.transform.rotation.x = orientation[0]
        aruco_frame_on_object.transform.rotation.y = orientation[1]
        aruco_frame_on_object.transform.rotation.z = orientation[2]
        aruco_frame_on_object.transform.rotation.w = orientation[3]
        self.transform_broadcaster.sendTransform(aruco_frame_on_object)

        #######################
        # 1. capture --> before aruco
        #######################
        # rotation = R.from_euler('y', 90, degrees=True)
        # rotation2 = R.from_euler('z', 90, degrees=True)
        # orientation = rotation2 * rotation * R.from_quat(orientation)
        # orientation = orientation.as_quat()
        # # orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
        #
        # aruco_frame_on_object = geometry_msgs.msg.TransformStamped()
        # aruco_frame_on_object.header.frame_id = self.base_link_frame
        # aruco_frame_on_object.child_frame_id = 'ee_target' # self.aruco_frame
        # aruco_frame_on_object.header.stamp = rospy.Time.now()
        # aruco_frame_on_object.transform.translation.x = position[0]
        # aruco_frame_on_object.transform.translation.y = position[1]
        # aruco_frame_on_object.transform.translation.z = position[2]
        # aruco_frame_on_object.transform.rotation.x = orientation[0]
        # aruco_frame_on_object.transform.rotation.y = orientation[1]
        # aruco_frame_on_object.transform.rotation.z = orientation[2]
        # aruco_frame_on_object.transform.rotation.w = orientation[3]
        # self.transform_broadcaster.sendTransform(aruco_frame_on_object)

        # Calculate required orientation for end effector (For elevator button)
        rotate_y_90 = R.from_quat(np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]))
        x_axis = np.array([1,0,0])
        rotate_angle = -np.arccos(np.dot(x_axis, -normal_dir))
        second_rotation = R.from_rotvec(rotate_angle * np.array([0, 0, 1]))
        orientation = second_rotation * rotate_y_90
        orientation = orientation.as_quat()

        # (For door)
        # rotate_x_90 = R.from_quat(np.array([-1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]))
        # orientation = rotate_x_90
        # orientation = orientation.as_quat()
        # print("Orientation: {}".format(orientation))


        # orientation = R.from_euler(np.array([3.142, 0, 0])).apply(orientation).to_quat()
        # orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
        before_aruco_position = before_position.copy()
        # before_aruco_position[0] -= 15 / 100                                    # offset for: 'wam/wrist_palm_link'
        # before_aruco_position[0] = before_aruco_position[0] - 10/100            # before aruco
        before_aruco_orientation = orientation
        # before_aruco_orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]) # perpendicular to the floor
        print(before_aruco_position)
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
        print("[trac-ik]   /wam/joint_states: ", wam_joint_states)
        if len(wam_joint_states) == 1:
            rospy.logwarn("IK solver failed ..")
            return

        self.send_barrett_to_joint_positions(joint_positions=wam_joint_states,
                                             barrett_arm_state="before arUco")
        time.sleep(10)

        #######################
        # 2. before aruco -> at aruco
        #######################

        at_aruco_position = after_position.copy()
        # at_aruco_position[0] -= 15 / 100                                         # offset for: 'wam/wrist_palm_link'
        at_aruco_orientation = orientation
        # at_aruco_orientation = np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)])  # perpendicular to the floor

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
        print("[trac-ik]   /wam/joint_states: ", wam_joint_states)
        if len(wam_joint_states) == 1:
            rospy.logwarn("IK solver failed ..")
            return

        self.send_barrett_to_joint_positions(joint_positions=wam_joint_states,
                                             barrett_arm_state="at arUco")
        time.sleep(10)

        #######################
        # 3. home position
        #######################

        self.send_barrett_to_joint_positions(self.home_joint_positions, barrett_arm_state='Home')

        # msg = Int8()
        # msg.data = # TODO
        # self.demo_sub_tasks_wam_pub.publish(msg)

        # msg = Bool()
        # self.is_summit_in_position = False
        # msg.data = self.is_summit_in_position
        # self.demo_sub_tasks_summit_pub.publish(msg)

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
    try: 
        main()
    except rospy.ROSInterruptException:
        pass
