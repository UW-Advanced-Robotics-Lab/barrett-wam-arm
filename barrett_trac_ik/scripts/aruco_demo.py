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
    CORRIDOR_DOOR_BUTTON    =  1  #: press the door button of the corridor
    ELEV_DOOR_BUTTON_CALL   =  2  #: press the elevator call button
    ELEV_DOOR_BUTTON_INSIDE =  3  #: press the floor button inside the elevator
    FAILED                  = -1  #: operation failed
    HOMING                  =  4  #: homing

class DEMO_STAGE_CHOREOGRAPHY(Enum):
    """ Demo Choreography Stage Enums

        Stage tokens for WAM arm choreography.

        @NOTE: [stage terms](https://medium.com/the-improv-blog/7-basic-stage-terms-for-improvisers-590870cf08a5)
    """
    OFF_STAGE           = 0 # "Initial/unknown stage"
    WINDING             = 1 # "Homing"
    WINGS               = 2 # "at Home"
    ON_STAGE            = 3 # "at Capture": at pre-capture joint positions
    IMPROVISATION       = 4 # "before pressing ArUco": live action based on zed-camera
    POST_IMPROVISATION  = 5 # "pressing ArUco": live action based on zed-camera

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
        WAM_REQUEST.ELEV_DOOR_BUTTON_INSIDE : {"aruco_x_dir_offset": 0, "aruco_y_dir_offset": -0.110, "button_press_norm_dist_factor": 0.115, "time_out_improvisation": 70, "time_out_post_improvisation": 50},
        WAM_REQUEST.ELEV_DOOR_BUTTON_CALL   : {"aruco_x_dir_offset": 0, "aruco_y_dir_offset": -0.177, "button_press_norm_dist_factor": 0.120, "time_out_improvisation": 50, "time_out_post_improvisation": 30},
        WAM_REQUEST.CORRIDOR_DOOR_BUTTON    : {"aruco_x_dir_offset": 0, "aruco_y_dir_offset":      0, "button_press_norm_dist_factor": 0.125, "time_out_improvisation": 50, "time_out_post_improvisation": 30},
        WAM_REQUEST.FAILED                  : {"button_press_norm_dist_factor": 0.125},
        WAM_REQUEST.HOMING                  : {"button_press_norm_dist_factor": 0.125},
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
        DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE           : 10,
        DEMO_STAGE_CHOREOGRAPHY.WINDING             : 50,
        DEMO_STAGE_CHOREOGRAPHY.WINGS               : 0,
        DEMO_STAGE_CHOREOGRAPHY.ON_STAGE            : 60,
        DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION       : 70, # default
        DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION  : 50, # default
    }
    
    #===============================#
    #  I N I T I A L I Z A T I O N  #
    #===============================#
    def __init__(self):
        #######################
        # Debugger:
        #######################
        self._verbose = False   # disable extra text prints
        self._logging_stage = "Init"    # stage tracking

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
        self._target_wam_request                = None
        self._target_zed_position               = None
        self._target_zed_position_after         = None
        self._target_zed_orientation            = None
        self._target_zed_orientation_after      = None
        
        self._stage_timeout_tick_100ms          = 0
        self._prev_transition_success           = True
        self._flag_fail_to_perform_the_request  = False

    #==================================#
    #  P U B L I C    F U N C T I O N  #
    #==================================#
    def run_100ms(self):
        """ 
        Update stage choreography per 100 [ms]

        @return success: False if there is any failure in the stage, else True
        """
        success = True
        new_stage = self._stage_check()

        if self._curr_stage is not new_stage:
            success &= self._stage_transition(new_stage = new_stage)
            if success:
                self._stage_timeout_tick_100ms = 0 # reset timeout ticks
            else:
                pass # re-attempt, till time-out
        
        # log success till this step:
        self._prev_transition_success = success

        # only perform current action, if no faults so far
        if success:
            success &= self._stage_action()

        return success

    #====================================#
    #  P R I V A T E    F U N C T I O N  #
    #====================================#
    def _format(self, info):
        return "[ STAGE: {} ] {}".format(self._logging_stage, info)

    def _print(self, info):
        if self._verbose:
            print(self._format(info=info))

    #######################
    # Staging Functions
    #######################
    def _stage_check(self):
        """ 
        Update parameters from input signals, and propose a new stage if necessary

        @return new_stage: new stage proposal if there is a need, else return the current stage
        """
        self._logging_stage = "CHECK"
        self._print(info="===== ===== ===== ===== ===== ===== ===== START:")

        ### Init ###
        new_stage = self._curr_stage
        wam_request = None
        position = None
        orientation = None

        ### Fetch ###
        with self._wam_lock:
            wam_request = self._wam_request
            self._wam_request = None # clear the cache
        with self._zed_lock:
            position    = self._zed_position
            orientation = self._zed_orientation
        
        self._print(info="r:{} p:{} o:{}".format(wam_request, position, orientation))

        ### Log ###
        # - Log AruCo Marker Zed Pose, when reaching towards ArUco Marker
        if self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
            if orientation is not None and position is not None:
                self._ros_log_object_pose(t=position, q=orientation)

        ### Determine A New Stage ###
        # - stages that are locked:
        if self._curr_stage in [
                # - wait for system auto booting
                None,
                DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE,
                # - wait for system homing
                DEMO_STAGE_CHOREOGRAPHY.WINDING,
                # - Do not interrupt the show, wait for completion!
                DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION,
                DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION
            ]:
            pass
        # - a new request while no show has been started (aka. only in homing position):
        elif self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
            if wam_request not in [None, WAM_REQUEST.FAILED]:
                if self._target_wam_request != wam_request: 
                    rospy.loginfo(self._format(info="New command ({}) has been accepted from SUMMIT!".format(wam_request)))
                    self._target_wam_request = wam_request
                    new_stage = DEMO_STAGE_CHOREOGRAPHY.ON_STAGE
                else:
                    rospy.logerr(self._format(info="Discard the repeated command received from SUMMIT!"))
            else:
                rospy.logwarn(self._format(info="No command has been received from SUMMIT!"))
        # re-try with new up-dated aruco position if the previously captured was not valid
        elif self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.ON_STAGE \
            and not self._prev_transition_success: 
            if position is not None and orientation is not None:
                self._target_zed_position    = position
                self._target_zed_orientation = orientation
                new_stage = DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION
    
        ### Timeout Overrides ###
        # Task specific tuned timeout:
        time_out_100ms = self._STAGE_TIME_OUT_100MS[self._curr_stage]
        if self._curr_stage in [
                # - Task specific time-out
                DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION,
                DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION
            ]:
                if self._target_wam_request is not None:
                    LUT = self._LUT_REQUEST_CONST_PARAMS[self._target_wam_request]
                    if "time_out_improvisation" in LUT and self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
                        time_out_100ms = LUT["time_out_improvisation"]
                    elif "time_out_post_improvisation" in LUT and self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION:
                        time_out_100ms = LUT["time_out_post_improvisation"]
        # timeout action:
        if self._stage_timeout_tick_100ms > time_out_100ms:

            if  self._prev_transition_success is not True:
                new_stage = DEMO_STAGE_CHOREOGRAPHY.WINDING # Abort, and go homing, if run failed, and timed-out
                self._flag_fail_to_perform_the_request = True
                pass # END

            # - [Time-out] Booted-on / Booted-off:
            elif      self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE:
                new_stage = DEMO_STAGE_CHOREOGRAPHY.WINDING # By Default, enter the winding stage to the home position
                pass # END

            # - [Time-out] Homing:
            elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.WINDING:
                # Note: Time-out buffer for the homing in progress
                new_stage = DEMO_STAGE_CHOREOGRAPHY.WINGS # By Default, enter the wings stage
                pass # END
            
            # - [Time-out] Initialized at Homing Position:
            elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
                # DO NOTHING, Note: No time-out for the homed stage
                pass # END
            
            # - [Time-out] Reached Pre-Captured Position:
            elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.ON_STAGE:
                if position is not None and orientation is not None:
                    self._target_zed_position    = position
                    self._target_zed_orientation = orientation
                    new_stage = DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION
                else:
                    rospy.logwarn(self._format(info="No ArUco Marker Found, waiting for target!"))
                pass # END
            
            # - [Time-out] Reached ArUco Marker:
            elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
                new_stage = DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION
                pass # Do Nothing
            
            # - [Time-out] Pressed ArUco Marker:
            elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION:
                new_stage = DEMO_STAGE_CHOREOGRAPHY.WINDING
                pass # Do Nothing
            
            # - Unknown:
            else:
                rospy.logerr(self._format(info="Invalid Current Stage."))
                pass # Do Nothing

        ### Homing CMD Override ###
        if wam_request is WAM_REQUEST.HOMING:
            new_stage = DEMO_STAGE_CHOREOGRAPHY.WINDING # entering winding to perform homing
        
        # - end:
        self._print(info="===== ===== ===== ===== ===== ===== ===== END. [return: {}]".format(new_stage))    
        return new_stage
    

    def _stage_transition(self, new_stage):
        """ 
        Things happen only once at the stage transition if successfully.

        @return success: False if transition failed, and `self._curr_stage` would not be overridden with the `new_stage` 
        """
        self._logging_stage = "TRANSITION: [{}]->[{}]".format(self._curr_stage, new_stage)
        rospy.logwarn(self._format(info="===== ===== ===== ===== ===== ===== ===== START:"))
        # init:
        success = True
        ### Action ###
        ###################################
        # -> Homing Override:
        if          new_stage is DEMO_STAGE_CHOREOGRAPHY.WINDING:
            ####################### BEGIN #######################
            # - Homing (only in this stage):
            self._send_barrett_to_joint_positions_non_block(self._LUT_CAPTURED_JOINT_POSITIONS[WAM_REQUEST.HOMING])
            #######################  END  #######################
        
        ###################################
        # -> Homed:
        elif          new_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
            ####################### BEGIN #######################
            if self._target_wam_request in [WAM_REQUEST.HOMING, None]:
                pass # do not report, if it was explicitly commanded to do homing
            else:
                # - Report Status (Once):
                status = WAM_STATUS[self._target_wam_request.name] 
                if self._flag_fail_to_perform_the_request:
                    status = WAM_STATUS.FAILED

                self._pub_aruco_demo_wam_status(status = status)

            # - Reset Caches:
            self._flag_fail_to_perform_the_request  = False
            self._target_wam_request                = None
            self._target_zed_position               = None
            self._target_zed_orientation            = None
            self._target_zed_position_after         = None
            self._target_zed_orientation_after      = None
            #######################  END  #######################

        ###################################
        # - Booted-on / Booted-off:
        elif      self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE:
            # -> Homing Once:
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.WINDING:
                ####################### BEGIN #######################
                pass # Do Nothing, make sure it only goes to winding
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
                    rospy.logwarn(self._format(
                        info="Can't find transform from {} to {}".format(self._WAM_JOINT_IDs["base_link_frame"], self._WAM_JOINT_IDs["camera_link_frame"])))
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

                    position += self._LUT_REQUEST_CONST_PARAMS[target]["aruco_x_dir_offset"] * tag_x_axis_dir
                    position += self._LUT_REQUEST_CONST_PARAMS[target]["aruco_y_dir_offset"] * tag_y_axis_dir

                    self._print(info="[Original] Normal Dir: {}".format(normal_dir))
                    # before_position = position.copy()
                    # position[2] += 0.025
                    factor_post_norm = self._LUT_REQUEST_CONST_PARAMS[target]["button_press_norm_dist_factor"]
                    before_position = position + 0.25 * normal_dir
                    after_position = position + factor_post_norm * normal_dir
                    self._print(info="[Offsetted] Before position: {}, After position: {}".format(before_position, position))


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
                    
                    self._print(info="[Result] Transformed Orientation: {}".format(orientation))


                    # orientation = R.from_euler(np.array([3.142, 0, 0])).apply(orientation).to_quat()
                    # orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
                    before_aruco_position = before_position.copy()
                    # before_aruco_position[0] -= 15 / 100                                    # offset for: 'wam/wrist_palm_link'
                    # before_aruco_position[0] = before_aruco_position[0] - 10/100            # before aruco
                    before_aruco_orientation = orientation
                    # before_aruco_orientation = np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]) # perpendicular to the floor
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
                        rospy.logwarn(self._format(info="IK solver failed !!"))
                        success = False
                    else:
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
                    rospy.logwarn(self._format(info="IK solver failed !!"))
                    success = False
                else:
                    self._send_barrett_to_joint_positions_non_block(wam_joint_states)
                #######################  END  #######################            
            else:
                success = False
        
        ###################################
        # - Pressed ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION:
            # -> Invoke to home, end of the choreography
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.WINDING:
                ####################### BEGIN #######################
                pass # do nothing
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
            rospy.logerr(self._format(info="Invalid Stage Transition From [{}] -x-> [{}]".format(self._curr_stage, new_stage)))
        
        # - end:
        rospy.logwarn(self._format(info="===== ===== ===== ===== ===== ===== ===== END. [return: {}]".format(success)))    
        return success

    def _stage_action(self):
        """ 
        Actions based on the current stage. [Note: Currently doing nothing]

        @return success: False if there is an error
        """
        self._logging_stage = "ACTION: [{}:{}]".format(self._curr_stage, self._stage_timeout_tick_100ms)
        self._print(info="===== ===== ===== ===== ===== ===== ===== START:")

        # - init:
        success = True

        # - Boot-on / Boot-off:
        if      self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE:
            pass # Do Nothing

        # - Homing:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.WINDING:
            pass # END
        
        # - Initialize at Homing Position:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.WINGS:
            pass # END
        
        # - Reaching Pre-Captured Position:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.ON_STAGE:
            pass # END
        
        # - Reaching ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
            pass # Do Nothing
        
        # - Pressing ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION:
            pass # Do Nothing
        
        # - Unknown:
        else:
            rospy.logerr(self._format("Invalid Stage Action."))
            pass # Do Nothing
        
        # - end:
        self._stage_timeout_tick_100ms += 1
        self._print(info="===== ===== ===== ===== ===== ===== ===== END. [return: {}]".format(success))    
        return success

    #######################
    # Helper Functions
    #######################
    def _pub_aruco_demo_wam_status(self, status):
        rospy.logwarn(self._format(info="[{}] *))) [trac-ik] /task_completion_flag_wam: [{}:{}]".format(self._curr_stage, status, status.value)))
        msg = Int8()
        msg.data = status.value # remap request to status
        self._publisher_status.publish(msg)
    
    def _send_barrett_to_joint_positions_non_block(self, joint_positions):
        self._print(info="[{}] ==> [trac-ik] /wam/joint_states: {}".format(self._curr_stage, joint_positions))
        # TODO: time-out based, we shall need a feedback loop by check states???
        msg = Float64MultiArray()
        msg.data = joint_positions
        # self._command_wam_arm_srv.publish(msg)
        self._command_wam_arm_srv.call(joint_positions)

    def _ros_log_object_pose(self, t, q):
        if self._verbose:
            # convert translation to [cm]
            t = t.copy() * 100

            rot = np.array(R.from_quat(q).as_dcm()).reshape(3, 3)
            r_vec, _ = cv2.Rodrigues(rot)
            r_vec = r_vec * 180 / np.pi
            r_vec = np.squeeze(np.array(r_vec)).reshape(-1)

            LOG_STR = "\n \
                Detected arUco marker: \n \
                position     [cm]: x:{:.2f}, y:{:.2f}, z:{:.2f} \n \
                orientation [deg]: x:{:.2f}, y:{:.2f}, z:{:.2f} \n \
            ".format(t[0], t[1], t[2], r_vec[0], r_vec[1], r_vec[2])
            rospy.loginfo(self._format(LOG_STR))

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

            self._print(info=" <-- [Summit] Request for {}".format(wam_request))
    
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
        self._print(info=" <-- [ZED] P{} R{}".format(position, orientation))

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
