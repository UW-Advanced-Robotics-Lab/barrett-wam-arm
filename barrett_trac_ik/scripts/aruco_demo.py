#! /usr/bin/env python
""" `aruco_demo.py`

    This script will perform aruco demo for manipulator interactions.

    @author: 
"""

# python libraries:
from __future__ import division

import os
import sys
import copy
import time
import math
import threading

from enum import Enum

# python 3rd party:
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy

# ours:
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #       ___           ___           ___           ___           ___       # #
# #      /  /\         /  /\         /  /\         /  /\         /  /\      # #
# #     /  /::\       /  /::\       /  /:/        /  /::\       /  /::\     # #
# #    /  /:/\:\     /  /:/\:\     /  /:/        /  /:/\:\     /  /:/\:\    # #
# #   /  /::\ \:\   /  /::\ \:\   /  /:/        /  /:/  \:\   /  /:/  \:\   # #
# #  /__/:/\:\_\:\ /__/:/\:\_\:\ /__/:/     /\ /__/:/ \  \:\ /__/:/ \__\:\  # #
# #  \__\/  \:\/:/ \__\/~|::\/:/ \  \:\    /:/ \  \:\  \__\/ \  \:\ /  /:/  # #
# #       \__\::/     |  |:|::/   \  \:\  /:/   \  \:\        \  \:\  /:/   # #
# #       /  /:/      |  |:|\/     \  \:\/:/     \  \:\        \  \:\/:/    # #
# #      /__/:/       |__|:|~       \  \::/       \  \:\        \  \::/     # #
# #      \__\/         \__\|         \__\/         \__\/         \__\/      # #
# #                                                                         # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#=======================#
#  D E F I N I T I O N  #
#=======================#
### ENUM ###
class WAM_REQUEST(Enum):
    CORRIDOR_DOOR_BUTTON    =  1  #: press the door button of the corridor
    ELEV_DOOR_BUTTON_CALL   =  2  #: press the elevator call button
    ELEV_DOOR_BUTTON_INSIDE =  3  #: press the floor button inside the elevator
    FAILED                  = -1  #: operation failed
    HOMING                  =  4  #: homing

class WAM_STATUS(Enum):
    # Will-publish:
    CORRIDOR_DOOR_BUTTON    =  101  #: press the door button of the corridor
    ELEV_DOOR_BUTTON_CALL   =  102  #: press the elevator call button
    ELEV_DOOR_BUTTON_INSIDE =  103  #: press the floor button inside the elevator
    FAILED                  = -101  #: operation failed
    HOMING                  =  104  #: homing

class DEMO_STAGE_CHOREOGRAPHY(Enum):
    """ Demo Choreography Stage Enums

        Stage tokens for WAM arm choreography.

        @NOTE: [stage terms](https://medium.com/the-improv-blog/7-basic-stage-terms-for-improvisers-590870cf08a5)
    """
    OFF_STAGE           = 0
    WINGS               = 1 # "at Home"
    ON_STAGE            = 2 # "at Capture": at pre-capture joint positions
    IMPROVISATION       = 3 # "before pressing ArUco": live action based on zed-camera
    POST_IMPROVISATION  = 4 # "pressing ArUco": live action based on zed-camera

class ArUcoDemo():
    #===================#
    #  C O N S T A N T  #
    #===================#
    ### Look up table for pre-calibrated joint-positions ###
    _LUT_CAPTURED_JOINT_POSITIONS = {
        WAM_REQUEST.ELEV_DOOR_BUTTON_INSIDE : [    
            0.00695301832677738,
            -0.4587789565136406,
            -0.002222416045176924,
            2.208318148967319,
            0.027892071199038658,
            -0.1788168083040264,
            -0.028431769350072793
        ],
        WAM_REQUEST.ELEV_DOOR_BUTTON_CALL : [
            0.00695301832677738,
            -0.4587789565136406,
            -0.002222416045176924,
            2.208318148967319,
            0.027892071199038658,
            -0.1788168083040264,
            -0.028431769350072793
        ],
        WAM_REQUEST.CORRIDOR_DOOR_BUTTON : [
            0.76,
            -0.4587789565136406,
            -0.002222416045176924,
            2.308318148967319,
            0.027892071199038658,
            -0.1788168083040264,
            -0.028431769350072793
        ],
        WAM_REQUEST.FAILED : [0, -1.25, 0, 3, 0, 0, 0],
        WAM_REQUEST.HOMING : [0, -1.25, 0, 3, 0, 0, 0]
    }

    _LUT_REQUEST_CONST_PARAMS = {
        WAM_REQUEST.CORRIDOR_DOOR_BUTTON    : None,
        WAM_REQUEST.ELEV_DOOR_BUTTON_CALL   : None,
        WAM_REQUEST.ELEV_DOOR_BUTTON_INSIDE : None,
        WAM_REQUEST.FAILED                  : None,
        WAM_REQUEST.HOMING                  : None,
    }

    _WAM_JOINT_IDs = {
            # WAM links / joints
            "base_link_frame"     : 'wam/base_link',
            "ee_link_frame"       : 'wam/wrist_palm_stump_link',  # 'wam/bhand/bhand_grasp_link' or 'wam/wrist_palm_link'
            "camera_link_frame"   : 'camera_frame',
            "aruco_frame_offset"  : 'aruco_frame_offset',
            # UNUSED:
            "forearm_link_frame"  : 'wam/forearm_link',
            "zed_link_frame"      : 'zed_camera_center',          # 'zed_camera_center' or 'camera_frame'
            "object_frame"        : 'object_frame', 
            "aruco_frame"         : 'aruco_frame',
    }

    _IK_SEED_STATE_IC = [
        # [0.00] * self._ik_solver.number_of_joints
        0.00010957005627754578, 0.8500968575130496, -0.00031928475261511213, 
        1.868559041954476, 0.0, -0.0006325693970662439, -0.00030823458564346445
    ]

    _STAGE_TIME_OUT_100MS = {
        DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE           : 50,
        DEMO_STAGE_CHOREOGRAPHY.WINGS               : 50,
        DEMO_STAGE_CHOREOGRAPHY.ON_STAGE            : 50,
        DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION       : 100,
        DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION  : 50,
    }
    
    #===============================#
    #  I N I T I A L I Z A T I O N  #
    #===============================#
    def __init__(self):
        #######################
        # ROS Topical
        #######################
        ### TF Tramsform Module ###
        self._tf_buffer             = tf2_ros.Buffer()
        self._listener              = tf2_ros.TransformListener(self._tf_buffer)
        self._transform_broadcaster = tf2_ros.TransformBroadcaster()

        ### Subscriber ###
        # - summit command subscriber
        self._subscriber_summit     = rospy.Subscriber("/task_completion_flag_summit", Int8, self._callback_upon_summit_cmd)
        # - zed camera feed with aruco pose: 
        self._object_pose_sub       = rospy.Subscriber("/aruco_single/pose", PoseStamped, self._callback_upon_zed_pose)

        ### Publisher ###
        # - publish WAM Choreography status
        self._publisher_status      = rospy.Publisher("/task_completion_flag_wam", Int8, queue_size=1)

        ### Service ###
        self._command_wam_arm_srv   = rospy.ServiceProxy("/wam/joint_move", JointMove)
        # self._command_wam_arm_srv = rospy.Publisher("/arm_position_controller/command", Float64MultiArray, queue_size=1)

        #######################
        # Other Modules
        #######################
        self._ik_solver = IK(self._WAM_JOINT_IDs["base_link_frame"], self._WAM_JOINT_IDs["ee_link_frame"])
        
        #######################
        # Initialization
        #######################
        # default stage at boot up:
        self._curr_stage = DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE
        
        # thread safe:
        self._wam_lock          = threading.RLock()
        self._wam_request       = None
        self._zed_lock          = threading.RLock()
        self._zed_position      = None
        self._zed_orientation   = None
        
        # non-thread cache placeholder
        self._target_wam_request        = None
        self._target_zed_position       = None
        self._target_zed_position_after = None
        self._target_zed_orientation    = None

        self._stage_timeout_tick_100ms = 0


    #==================================#
    #  P U B L I C    F U N C T I O N  #
    #==================================#
    def run_100ms(self):
        """ 
        @brief: update stage choreography per 100 [ms]
        """
        new_stage = self._stage_check()
        print("> [{}] -> [{}]".format(self._curr_stage, new_stage))
        if self._curr_stage is not new_stage:
            self._stage_transition(new_stage = new_stage)
        self._stage_action()

    #====================================#
    #  P R I V A T E    F U N C T I O N  #
    #====================================#
    #######################
    # Staging Functions
    #######################
    def _stage_check(self):
        """ 
        @brief: check stage changes and determine the new stage
        """
        ### Init ###
        new_stage = self._curr_stage
        wam_request = None
        position = None
        orientation = None

        ### Fetch ###
        with self._wam_lock:
            wam_request = self._wam_request
            self._wam_request = None
        with self._zed_lock:
            position    = self._zed_position
            orientation = self._zed_orientation
        
        print("[Info] r:{} p:{} o:{}".format(wam_request, position, orientation))

        ### Log ###
        # - Log AruCo Marker Zed Pose, when reaching towards ArUco Marker
        if self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
            if orientation is not None and position is not None:
                self._ros_log_object_pose(t=position, q=orientation)

        ### Determine A New Stage ###
        if self._curr_stage in [
                # - wait for system auto booting
                None,
                DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE,
                # - Do not interrupt the show, wait for completion!
                DEMO_STAGE_CHOREOGRAPHY.ON_STAGE,
                DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION,
                DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION
            ]:
            pass
        # - a new request while no show has been started (aka. in homing position):
        elif self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
            if wam_request not in [None, WAM_REQUEST.FAILED]:
                self._target_wam_request = wam_request
                new_stage = DEMO_STAGE_CHOREOGRAPHY.ON_STAGE
            else:
                # rospy.logerr("Invalid WAM Request from SUMMIT.")
                print("- No command yet!")

        ### Timeout Overrides ###
        if self._stage_timeout_tick_100ms > self._STAGE_TIME_OUT_100MS[self._curr_stage]:

            # - [Time-out] Booted-on / Booted-off:
            if      self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE:
                new_stage = DEMO_STAGE_CHOREOGRAPHY.WINGS # By Default, enter the wings stage
                pass # END

            # - [Time-out] Initialized at Homing Position:
            elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
                # DO NOTHING, Note: No time-out for the homing stage
                pass # END
            
            # - [Time-out] Reached Pre-Captured Position:
            elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.ON_STAGE:
                if position is not None and orientation is not None:
                    self._target_zed_position    = position
                    self._target_zed_orientation = orientation
                    new_stage = DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION
                else:
                    rospy.logwarn("No ArUco Marker Found, waiting for target!")
                pass # END
            
            # - [Time-out] Reached ArUco Marker:
            elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
                new_stage = DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION
                pass # Do Nothing
            
            # - [Time-out] Pressed ArUco Marker:
            elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION:
                new_stage = DEMO_STAGE_CHOREOGRAPHY.WINGS
                pass # Do Nothing
            
            # - Unknown:
            else:
                rospy.logerr("Invalid Current Stage.")
                pass # Do Nothing

        ### Homing CMD Override ###
        if wam_request is WAM_REQUEST.HOMING:
            new_stage = DEMO_STAGE_CHOREOGRAPHY.WINGS
        
        return new_stage
    

    def _stage_transition(self, new_stage):
        """ 
        @brief: actions based on stage transition
        """
        success = True
        self._stage_timeout_tick_100ms = 0 # reset timeout ticks
        ### Action ###
        ###################################
        # -> Homing Override:
        if          new_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
            ####################### BEGIN #######################
            self._send_barrett_to_joint_positions_non_block(self._LUT_CAPTURED_JOINT_POSITIONS[WAM_REQUEST.HOMING])
            #######################  END  #######################
        ###################################
        # - Booted-on / Booted-off:
        elif      self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE:
            # -> Homing Once:
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
                ####################### BEGIN #######################
                self._send_barrett_to_joint_positions_non_block(self._LUT_CAPTURED_JOINT_POSITIONS[WAM_REQUEST.HOMING])
                #######################  END  #######################
            else:
                success = False

        ###################################
        # - Initialized at Homing Position:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
            # -> Invoke Capturing Position Once:
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.ON_STAGE:
                ####################### BEGIN #######################
                self._send_barrett_to_joint_positions_non_block(self._LUT_CAPTURED_JOINT_POSITIONS[self._target_wam_request])
                #######################  END  #######################
            else:
                success = False
        
        ###################################
        # - Reached Pre-Captured Position:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.ON_STAGE:
            # -> Invoke to AruCo Marker
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
                ####################### BEGIN #######################
                # - cache pose locally:
                position = self._target_zed_position
                orientation = self._target_zed_orientation
                target = self._target_wam_request

                ### pub tf ###
                # orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
                # - TF:
                aruco_frame_wrt_camera_frame = geometry_msgs.msg.TransformStamped()
                aruco_frame_wrt_camera_frame.header.frame_id = self._WAM_JOINT_IDs["camera_link_frame"]
                aruco_frame_wrt_camera_frame.child_frame_id = self._WAM_JOINT_IDs["aruco_frame"]
                aruco_frame_wrt_camera_frame.header.stamp = rospy.Time.now()
                aruco_frame_wrt_camera_frame.transform.translation.x = position[0]
                aruco_frame_wrt_camera_frame.transform.translation.y = position[1]
                aruco_frame_wrt_camera_frame.transform.translation.z = position[2]
                aruco_frame_wrt_camera_frame.transform.rotation.x = orientation[0]
                aruco_frame_wrt_camera_frame.transform.rotation.y = orientation[1]
                aruco_frame_wrt_camera_frame.transform.rotation.z = orientation[2]
                aruco_frame_wrt_camera_frame.transform.rotation.w = orientation[3]
                self._transform_broadcaster.sendTransform(aruco_frame_wrt_camera_frame)

                ### Transforms from zed cam feed ###
                time.sleep(3)
                try:

                    # pose
                    object_in_camera_frame_msg = PoseStamped()
                    # object_in_camera_frame_msg.header.frame_id = self._WAM_JOINT_IDs["camera_link_frame"]
                    object_in_camera_frame_msg.pose.position.x = position[0]
                    object_in_camera_frame_msg.pose.position.y = position[1]
                    object_in_camera_frame_msg.pose.position.z = position[2]
                    object_in_camera_frame_msg.pose.orientation.w = orientation[3]
                    object_in_camera_frame_msg.pose.orientation.x = orientation[0]
                    object_in_camera_frame_msg.pose.orientation.y = orientation[1]
                    object_in_camera_frame_msg.pose.orientation.z = orientation[2]

                    ''' object_T_world = object_T_zed * zed_T_world '''
                    # grabbing transformation from camera to the base from the TF node
                    # zed_T_world
                    camera_to_world = self._tf_buffer.lookup_transform(self._WAM_JOINT_IDs["base_link_frame"], self._WAM_JOINT_IDs["camera_link_frame"], rospy.Time(0))
                    # object_T_world
                    object_to_world = tf2_geometry_msgs.do_transform_pose(object_in_camera_frame_msg, camera_to_world)

                    position = np.array([   object_to_world.pose.position.x,
                                            object_to_world.pose.position.y,
                                            object_to_world.pose.position.z])

                    orientation = np.array([object_to_world.pose.orientation.x,
                                            object_to_world.pose.orientation.y,
                                            object_to_world.pose.orientation.z,
                                            object_to_world.pose.orientation.w])

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logwarn("Can't find transform from {} to {}".format(self._WAM_JOINT_IDs["base_link_frame"], self._WAM_JOINT_IDs["camera_link_frame"]))
                    success = False

                ### Tranform to joint positions ###
                if success:
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

                    if target is WAM_REQUEST.ELEV_DOOR_BUTTON_CALL:
                        # NOTE: Next two lines are for the elevator (outside) only (comment out for other locations)
                        position += -0.017 * tag_x_axis_dir
                        position += -0.177 * tag_y_axis_dir
                    elif target is WAM_REQUEST.ELEV_DOOR_BUTTON_INSIDE:    
                        # NOTE: Next two lines are for the elevator (inside) only (comment out for other locations)
                        position += 0.02 * tag_y_axis_dir
                        position += -0.11 * tag_x_axis_dir

                    print("Normal Dir: {}".format(normal_dir))
                    # before_position = position.copy()
                    # position[2] += 0.025
                    before_position = position + 0.25 * normal_dir
                    after_position = position + 0.125 * normal_dir
                    print("Befpre position: {}, After position: {}".format(before_position, position))


                    ### modify marker value in sim ###
                    # position = np.array([0.5, 0, 0.85])
                    # orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]) # perpendicular to the floor
                    # self._ros_log_object_pose(t=position, q=orientation)

                    ### pub tf ###
                    # - tf
                    aruco_frame_wrt_camera_frame = geometry_msgs.msg.TransformStamped()
                    aruco_frame_wrt_camera_frame.header.frame_id = self._WAM_JOINT_IDs["base_link_frame"]
                    aruco_frame_wrt_camera_frame.child_frame_id = self._WAM_JOINT_IDs["aruco_frame_offset"] # self._WAM_JOINT_IDs["aruco_frame"]
                    aruco_frame_wrt_camera_frame.header.stamp = rospy.Time.now()
                    aruco_frame_wrt_camera_frame.transform.translation.x = before_position[0]
                    aruco_frame_wrt_camera_frame.transform.translation.y = before_position[1]
                    aruco_frame_wrt_camera_frame.transform.translation.z = before_position[2]
                    aruco_frame_wrt_camera_frame.transform.rotation.x = orientation[0]
                    aruco_frame_wrt_camera_frame.transform.rotation.y = orientation[1]
                    aruco_frame_wrt_camera_frame.transform.rotation.z = orientation[2]
                    aruco_frame_wrt_camera_frame.transform.rotation.w = orientation[3]
                    self._transform_broadcaster.sendTransform(aruco_frame_wrt_camera_frame)

                    ###############################
                    # 1. capture --> before aruco #
                    ###############################
                    ### DEBUGGING CODE ###
                    # TODO: note: position of the arm before pressing aruco marker, for DEBUGGING
                    # rotation = R.from_euler('y', 90, degrees=True)
                    # rotation2 = R.from_euler('z', 90, degrees=True)
                    # orientation = rotation2 * rotation * R.from_quat(orientation)
                    # orientation = orientation.as_quat()
                    # # orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
                    #
                    # aruco_frame_wrt_camera_frame = geometry_msgs.msg.TransformStamped()
                    # aruco_frame_wrt_camera_frame.header.frame_id = self._WAM_JOINT_IDs["base_link_frame"]
                    # aruco_frame_wrt_camera_frame.child_frame_id = 'ee_target' # self._WAM_JOINT_IDs["aruco_frame"]
                    # aruco_frame_wrt_camera_frame.header.stamp = rospy.Time.now()
                    # aruco_frame_wrt_camera_frame.transform.translation.x = position[0]
                    # aruco_frame_wrt_camera_frame.transform.translation.y = position[1]
                    # aruco_frame_wrt_camera_frame.transform.translation.z = position[2]
                    # aruco_frame_wrt_camera_frame.transform.rotation.x = orientation[0]
                    # aruco_frame_wrt_camera_frame.transform.rotation.y = orientation[1]
                    # aruco_frame_wrt_camera_frame.transform.rotation.z = orientation[2]
                    # aruco_frame_wrt_camera_frame.transform.rotation.w = orientation[3]
                    # self._transform_broadcaster.sendTransform(aruco_frame_wrt_camera_frame)

                    ### Orientation ###
                    # - Calculate required orientation for end effector:
                    # - (For elevator button)
                    if target in [WAM_REQUEST.ELEV_DOOR_BUTTON_CALL, WAM_REQUEST.ELEV_DOOR_BUTTON_INSIDE]:
                        rotate_y_90 = R.from_quat(np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]))
                        x_axis = np.array([1,0,0])
                        rotate_angle = -np.arccos(np.dot(x_axis, -normal_dir))
                        second_rotation = R.from_rotvec(rotate_angle * np.array([0, 0, 1]))
                        orientation = second_rotation * rotate_y_90
                        orientation = orientation.as_quat()
                    # - (For door)
                    elif target is WAM_REQUEST.CORRIDOR_DOOR_BUTTON:
                        rotate_x_90 = R.from_quat(np.array([-1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]))
                        orientation = rotate_x_90
                        orientation = orientation.as_quat()
                        print("Orientation: {}".format(orientation))


                    # orientation = R.from_euler(np.array([3.142, 0, 0])).apply(orientation).to_quat()
                    # orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
                    before_aruco_position = before_position.copy()
                    # before_aruco_position[0] -= 15 / 100                                    # offset for: 'wam/wrist_palm_link'
                    # before_aruco_position[0] = before_aruco_position[0] - 10/100            # before aruco
                    before_aruco_orientation = orientation
                    # before_aruco_orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]) # perpendicular to the floor
                    print(before_aruco_position)
                    wam_joint_states = self._ik_solver.get_ik(
                        self._IK_SEED_STATE_IC,
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
                    if len(wam_joint_states) == 1:
                        rospy.logwarn("IK solver failed ..")
                        success = False
                    else:
                        print("[trac-ik]   /wam/joint_states: ", wam_joint_states)
                        self._send_barrett_to_joint_positions_non_block(wam_joint_states)

                    ### Record ###
                    self._target_zed_position_after = after_position # cached towards the next stage
                    self._target_zed_orientation_after = orientation
                #######################  END  #######################
            else:
                success = False
        
        ###################################
        # - Reached ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
            # -> Invoke to press AruCo Marker
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION:
                ####################### BEGIN #######################
                # - cache pose locally:
                orientation = self._target_zed_orientation_after
                after_position = self._target_zed_position_after

                ### Compute ArUco Pressing Action:
                at_aruco_position = after_position.copy()
                # at_aruco_position[0] -= 15 / 100                                         # offset for: 'wam/wrist_palm_link'
                at_aruco_orientation = orientation
                # at_aruco_orientation = np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)])  # perpendicular to the floor

                wam_joint_states = self._ik_solver.get_ik(self._IK_SEED_STATE_IC,
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
                if len(wam_joint_states) == 1:
                    rospy.logwarn("IK solver failed ..")
                    success = False
                else:
                    print("[trac-ik]   /wam/joint_states: ", wam_joint_states)
                    self._send_barrett_to_joint_positions_non_block(wam_joint_states)
                #######################  END  #######################            
            else:
                success = False
        
        ###################################
        # - Pressed ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION:
            # -> Invoke to home, end of the choreography
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
                ####################### BEGIN #######################
                # - HOMING:
                self._send_barrett_to_joint_positions_non_block(self._LUT_CAPTURED_JOINT_POSITIONS[WAM_REQUEST.HOMING])

                # - Report Status Once:
                if self.is_barrett_capture:
                    msg = Int8()
                    msg.data = WAM_STATUS[self._target_wam_request.name].value # remap request to status
                    self.demo_sub_tasks_wam_pub.publish(msg)

                # - Reset Caches:
                self._target_wam_request        = None
                self._target_zed_position       = None
                self._target_zed_orientation    = None
                self._target_zed_position_after       = None
                self._target_zed_orientation_after    = None
                #######################  END  #######################
            else:
                success = False
        # - Unknown:
        else:
            success = False

        ### Update ###
        if success:
            # - enter the new stage
            self._curr_stage = new_stage
        else:
            # - stuck in stage transition
            rospy.logerr("Invalid Stage Transition From [{}] -x-> [{}]".format(self._curr_stage, new_stage))
            
        return success

    def _stage_action(self):
        """ 
        @brief: actions based on the current stage
        """
        # - Boot-on / Boot-off:
        if      self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE:
            pass # Do Nothing

        # - Initialize at Homing Position:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
            pass # END
        
        # - Reaching Pre-Captured Position:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.ON_STAGE:
            # rospy.loginfo("Summit is in position for Barrett Arm operations! [REQUEST: {}]".format(self.wam_request))
            pass # END
        
        # - Reaching ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
            pass # Do Nothing
        
        # - Pressing ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION:
            pass # Do Nothing
        
        # - Unknown:
        else:
            rospy.logerr("Invalid Stage Transition.")
            pass # Do Nothing
        
        self._stage_timeout_tick_100ms += 1
    
    #######################
    # Helper Functions
    #######################
    def _send_barrett_to_joint_positions_non_block(self, joint_positions):
        rospy.loginfo("[WAM] commanding to --> {} ..".format(self._curr_stage))
        # TODO: time-out based, we shall need a feedback loop by check states???
        msg = Float64MultiArray()
        msg.data = joint_positions
        # self._command_wam_arm_srv.publish(msg)
        self._command_wam_arm_srv.call(joint_positions)

    def _ros_log_object_pose(self, t, q):
        # convert translation to [cm]
        t = t.copy() * 100

        rot = np.array(R.from_quat(q).as_dcm()).reshape(3, 3)
        r_vec, _ = cv2.Rodrigues(rot)
        r_vec = r_vec * 180 / np.pi
        r_vec = np.squeeze(np.array(r_vec)).reshape(-1)

        rospy.loginfo('')
        rospy.loginfo('Detected arUco marker:')
        rospy.loginfo('position     [cm]: x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(t[0], t[1], t[2]))
        rospy.loginfo('orientation [deg]: x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(r_vec[0], r_vec[1], r_vec[2]))


    #######################
    # Callback Functions
    #######################
    def _callback_upon_summit_cmd(self, is_summit_in_position_msg):
        if is_summit_in_position_msg.data:
            ### Pre-process ###
            wam_request = WAM_REQUEST.FAILED
            try: 
                wam_request = WAM_REQUEST(is_summit_in_position_msg.data)
            except ValueError: # treat out-of-scope as failure
                wam_request = WAM_REQUEST.FAILED
                rospy.logwarn("Invalid Request [SUMMIT:{}]!".format(is_summit_in_position_msg.data))
            
            ### Capture ###
            with self._wam_lock:
                self._wam_request = wam_request

            print(" [Summit] > Request for {}".format(wam_request))
    
    def _callback_upon_zed_pose(self, object_in_camera_frame_msg):
        ### Init. ###
        x,y,z = 0.0, 0.0, 0.0
        
        ### Pre-process ###
        x = object_in_camera_frame_msg.pose.position.x  # /10
        y = object_in_camera_frame_msg.pose.position.y  # /10
        z = object_in_camera_frame_msg.pose.position.z  # /10
        # y += 13/100 # offset for: 'wam/wrist_palm_link'
        position = np.array([x, y, z])
        w = object_in_camera_frame_msg.pose.orientation.w
        x = object_in_camera_frame_msg.pose.orientation.x
        y = object_in_camera_frame_msg.pose.orientation.y
        z = object_in_camera_frame_msg.pose.orientation.z
        # orientation = np.array([w, x, y, z])
        orientation = np.array([x, y, z, w])
        # orientation = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0])
        print(" [ZED] > P{} R{}".format(position, orientation))

        ### Capture ###
        with self._zed_lock:
            self._zed_position    = position
            self._zed_orientation = orientation


#===========#
#  M A I N  #
#===========#
def main():

    rospy.init_node('barrett_trac_ik', anonymous=True)
    demo = ArUcoDemo()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        demo.run_100ms()
        rate.sleep()

if __name__ == '__main__':
    try: 
        main()
    except rospy.ROSInterruptException:
        pass
