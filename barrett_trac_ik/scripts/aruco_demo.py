#! /usr/bin/env python
""" `aruco_demo.py`

    This script will perform aruco demo for manipulator interactions.

    @author: 
"""

# python libraries:
from __future__ import division
from fnmatch import translate
from ntpath import join

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

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3, Quaternion, Transform, Pose
from std_msgs.msg import Bool, Float64MultiArray, Int8, Header

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
    RE_IMPROVISATION    = 6 # back to "before pressing ArUco", release pressure

class ArUcoDemo():
    #===================#
    #  C O N S T A N T  #
    #===================#
    ### Config ###
    _JOINT_POSITION_TOL = 0.020
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
        WAM_REQUEST.FAILED : [0, -1.75, 0, 3, 0, 0, 0],
        WAM_REQUEST.HOMING : [0, -1.75, 0, 3, 0, 0, 0]
    }

    _LUT_REQUEST_CONST_PARAMS = {
        # TODO: tuning needed
        WAM_REQUEST.ELEV_DOOR_BUTTON_INSIDE : {"aruco_x_dir_offset":    0.005, "aruco_y_dir_offset":  -0.200, "button_press_norm_dist_factor": 0.002, "button_delta": 0.030,  "time_out_improvisation": 80, "time_out_post_improvisation": 25},
        WAM_REQUEST.ELEV_DOOR_BUTTON_CALL   : {"aruco_x_dir_offset":    -0.007, "aruco_y_dir_offset":   -0.212, "button_press_norm_dist_factor": 0.013, "button_delta": 0.030,  "time_out_improvisation": 80, "time_out_post_improvisation": 20},
        WAM_REQUEST.CORRIDOR_DOOR_BUTTON    : {"aruco_x_dir_offset":    0, "aruco_y_dir_offset":   -0.200, "button_press_norm_dist_factor": +0.001, "button_delta": 0.100,  "time_out_improvisation": 80, "time_out_post_improvisation": 25}, # calibrate on March 09
        # WAM_REQUEST.CORRIDOR_DOOR_BUTTON    : {"aruco_x_dir_offset":    0, "aruco_y_dir_offset":      0, "button_press_norm_dist_factor": -0.012, "time_out_improvisation": 120, "time_out_post_improvisation": 40}, # Button press downstair
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
            "zed_link_frame"      : 'zed_camera_center',          # 'zed_link_frame' or 'camera_frame'
            # "object_frame"        : 'object_frame', 
            #-- add-ons:
            "aruco_frame"         : 'aruco_frame',
            # "aruco_frame_offset_before": 'aruco_frame_offset_before',
            # "aruco_frame_offset_after": 'aruco_frame_offset_after',
    }

    # _ROS_TOPICAL = {
    #     "subscriber" : {
    #         "summit-flag" : "/task_completion_flag_summit",
    #         "aruco-pose" : "/aruco_single/pose",
    #         "wam-joint-states" : "/wam/joint_states",
    #     },
    #     "publisher" : {
    #         "wam-flag" : "/task_completion_flag_wam",
    #     },
    #     "service" : {
    #         "wam-move" : "/wam/joint_move"
    #     }
    # }

    _IK_SEED_STATE_IC = [
        # [0.00] * self._ik_solver.number_of_joints
        0.00010957005627754578, 0.8500968575130496, -0.00031928475261511213, 
        1.868559041954476, 0.0, -0.0006325693970662439, -0.00030823458564346445
    ]

    _STAGE_TIME_OUT_100MS = {
        DEMO_STAGE_CHOREOGRAPHY.OFF_STAGE           : 10,
        DEMO_STAGE_CHOREOGRAPHY.WINDING             : 20, # pub completion flag after winding --> wings
        DEMO_STAGE_CHOREOGRAPHY.WINGS               : 10,
        DEMO_STAGE_CHOREOGRAPHY.ON_STAGE            : 50,#40,
        DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION       : 50,#70, # default
        DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION  : 50,#50, # default
        DEMO_STAGE_CHOREOGRAPHY.RE_IMPROVISATION    : 15,
    }
    
    # Measurements of ZED to forearm link in world coords. TODO: make it some sort of const. 
    _CONST_OFFSET_TO_LET_ZED_LENS_X_AXIS = -135 / 1000  # [m] height from zed to forearm link
    _CONST_OFFSET_TO_LET_ZED_LENS_Y_AXIS = -171 / 1000  # distance from zed to forearm link. TODO: this is a rough measurement.
    _CONST_OFFSET_TO_LET_ZED_LENS_Z_AXIS = 65 / 1000    # camera center to left lens.

    _CONST_POSES_IN_WORLD_COORD = {
        "zed_link_frame": TransformStamped(
            header=Header(
                frame_id=_WAM_JOINT_IDs["forearm_link_frame"]
            ), 
            child_frame_id=_WAM_JOINT_IDs["zed_link_frame"],
            transform=Transform(
                translation = Vector3(
                    _CONST_OFFSET_TO_LET_ZED_LENS_X_AXIS,
                    _CONST_OFFSET_TO_LET_ZED_LENS_Y_AXIS,
                    0
                ), 
                rotation = Quaternion(
                    1 / np.sqrt(2),
                    0,
                    0,
                    1 / np.sqrt(2)
                )
            )
        ),
        "camera_link_frame": TransformStamped(
            header=Header(
                frame_id=_WAM_JOINT_IDs["forearm_link_frame"]
            ), 
            child_frame_id=_WAM_JOINT_IDs["camera_link_frame"],
            transform=Transform(
                translation = Vector3(
                    _CONST_OFFSET_TO_LET_ZED_LENS_X_AXIS,
                    _CONST_OFFSET_TO_LET_ZED_LENS_Y_AXIS,
                    _CONST_OFFSET_TO_LET_ZED_LENS_Z_AXIS,
                ), 
                rotation = Quaternion(
                    0.5,
                    0.5,
                    -0.5,
                    0.5
                )
            )
        ),
    }

    #===============================#
    #  I N I T I A L I Z A T I O N  #
    #===============================#
    def __init__(self):
        #######################
        # Debugger:
        #######################
        self._verbose = True   # disable extra text prints
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
        # - wam joint-state callbacks
        self.joint_states_sub       = rospy.Subscriber("/wam/joint_states", JointState, self._callback_upon_wam_joint_states)

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
        self._zed_framestamp     = None
        self._zed_position      = None
        self._zed_orientation   = None
        self._wam_joint_state_lock  = threading.RLock()
        self._wam_joint_position    = None
        self._wam_time_stamp        = None
        
        # non-thread cache placeholder
        self._target_wam_request                = None
        self._target_zed_position               = None
        self._target_aruco_position_after       = None
        self._target_zed_orientation            = None
        self._target_aruco_orientation_after    = None
        self._target_aruco_pose_before          = None
        self._target_aruco_pose_after           = None
        
        self._stage_timeout_tick_100ms          = 0
        self._prev_transition_success           = True
        self._flag_fail_to_perform_the_request  = False
        
        self._target_wam_joints                 = None
        self._wam_joint_position_previous       = None

        self._demo_stage_begin_time             = rospy.Time.now()

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
                # log time:
                delta_time = rospy.Time.now() - self._demo_stage_begin_time
                self._print(info=" > Stage Ellapsed {} s ".format(delta_time.to_sec()))
                # reset:
                self._stage_timeout_tick_100ms = 0 # reset timeout ticks
                self._demo_stage_begin_time = rospy.Time.now() # capture time
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
        wam_joint_position = None
        position_reached = False
        zed_aruco_framestamp = None

        ### Fetch ###
        with self._wam_lock:
            wam_request = self._wam_request
            self._wam_request = None # clear the cache
        with self._zed_lock:
            zed_aruco_framestamp = self._zed_framestamp
            position    = self._zed_position
            orientation = self._zed_orientation
        with self._wam_joint_state_lock:
            wam_joint_position  = self._wam_joint_position
        
        self._print(info="r:{} p:{} o:{}".format(wam_request, position, orientation))
        self._print(info="joint position:{}".format(wam_joint_position))

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
                DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION,
                DEMO_STAGE_CHOREOGRAPHY.RE_IMPROVISATION
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
            if zed_aruco_framestamp is not None and \
                zed_aruco_framestamp > self._demo_stage_begin_time: # make sure only the aruco after staged is trusted
                    self._target_zed_position    = position
                    self._target_zed_orientation = orientation
                    new_stage = DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION
    
        ### Position Reached Overrides ###
        if self._target_wam_joints and self._wam_joint_position_previous is not None:
            delta_change = np.linalg.norm(np.array(wam_joint_position) - np.array(self._wam_joint_position_previous))
            delta = np.linalg.norm(np.array(wam_joint_position) - np.array(self._target_wam_joints))
            # - Skip the position, iff delta change and delta is smaller than a specified TOL (can be different threshold)
            if delta <= self._JOINT_POSITION_TOL and delta_change <= self._JOINT_POSITION_TOL:
                position_reached = True
                rospy.logwarn(self._format(info="Position Reached, Skip !!!!"))
            else:
                rospy.logwarn(self._format(info="Position NOT Reached, NOT Skip !!!! {}".format(delta)))
        ## log joint position: 
        self._wam_joint_position_previous = wam_joint_position

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
        if self._stage_timeout_tick_100ms > time_out_100ms or position_reached:

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
                new_stage = DEMO_STAGE_CHOREOGRAPHY.RE_IMPROVISATION
                pass # Do Nothing
            
            # - [Time-out] Released ArUco Marker:
            elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.RE_IMPROVISATION:
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
            self._target_wam_joints = (self._LUT_CAPTURED_JOINT_POSITIONS[WAM_REQUEST.HOMING])
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
            self._target_aruco_pose_before          = None
            self._target_aruco_pose_after           = None
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
                self._target_wam_joints = (self._LUT_CAPTURED_JOINT_POSITIONS[self._target_wam_request])
                #######################  END  #######################
            else:
                success = False
        
        ###################################
        # - Reached Pre-Captured Position:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.ON_STAGE:
            # -> Invoke to AruCo Marker
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
                ####################### BEGIN #######################
                ### INIT: ###
                # - cache pose locally:
                position = self._target_zed_position
                orientation = self._target_zed_orientation
                target = self._target_wam_request
                # - reset placeholders:
                self._target_aruco_pose_before          = None
                self._target_aruco_pose_after           = None

                with self._wam_joint_state_lock:
                    time_now = self._wam_time_stamp  # feed wam time stamp, for tf time matching
                
                ### COMPUTE ###
                # - init:
                X_AXIS = np.array([1,0,0])
                Y_AXIS = np.array([0,1,0])
                Z_AXIS = np.array([0,0,1])
                # - compute axis:
                normal_dir = R.from_quat(orientation).apply(Z_AXIS)
                tag_x_axis_dir = R.from_quat(orientation).apply(X_AXIS)
                tag_y_axis_dir = R.from_quat(orientation).apply(Y_AXIS)
                # - apply offset:
                position_offsetted = position \
                    + self._LUT_REQUEST_CONST_PARAMS[target]["aruco_x_dir_offset"] * tag_x_axis_dir \
                    + self._LUT_REQUEST_CONST_PARAMS[target]["aruco_y_dir_offset"] * tag_y_axis_dir

                factor_post_norm = self._LUT_REQUEST_CONST_PARAMS[target]["button_press_norm_dist_factor"]
                delta_button = self._LUT_REQUEST_CONST_PARAMS[target]["button_delta"]
                before_position     = position_offsetted + (factor_post_norm + delta_button) * normal_dir   # - aiming
                after_position      = position_offsetted + factor_post_norm * normal_dir            # - pressing the button

                # - print:
                self._print(info="[Aruco Mark ] Original Position: {}, XY-Offsetted Position: {}".format(position, position_offsetted))
                self._print(info="[Offsetted-Z]   Before position: {},        After position: {}".format(before_position, after_position))


                ### TF PUB ###
                # aruco_frame_wrt_camera_frame_before = TransformStamped(
                #     header=Header(
                #         frame_id=self._WAM_JOINT_IDs["camera_link_frame"],
                #         stamp = time_now
                #     ), 
                #     child_frame_id=self._WAM_JOINT_IDs["aruco_frame_offset_before"],
                #     transform = Transform(
                #         translation = self._array_to_vector3(before_position),
                #         rotation = self._array_to_quaternion(orientation)
                #     )
                # )
                # aruco_frame_wrt_camera_frame_after = TransformStamped(
                #     header=Header(
                #         frame_id=self._WAM_JOINT_IDs["camera_link_frame"],
                #         stamp = time_now
                #     ), 
                #     child_frame_id=self._WAM_JOINT_IDs["aruco_frame_offset_after"],
                #     transform = Transform(
                #         translation = self._array_to_vector3(after_position),
                #         rotation = self._array_to_quaternion(orientation)
                #     )
                # )

                # - init local Poses:
                aruco_frame_wrt_camera_frame = TransformStamped(
                    header=Header(
                        frame_id=self._WAM_JOINT_IDs["camera_link_frame"],
                        stamp = time_now
                    ), 
                    child_frame_id=self._WAM_JOINT_IDs["aruco_frame"],
                    transform = Transform(
                        translation = self._array_to_vector3(position),
                        rotation = self._array_to_quaternion(orientation)
                    )
                )
                aruco_frame_wrt_camera_frame_before_pose = PoseStamped(
                    pose = Pose(
                        position = self._array_to_vector3(before_position),
                        orientation = self._array_to_quaternion(orientation)
                    )
                )
                aruco_frame_wrt_camera_frame_after_pose = PoseStamped(
                    pose = Pose(
                        position = self._array_to_vector3(after_position),
                        orientation = self._array_to_quaternion(orientation)
                    )
                )

                # - pub current transformation frame
                self._transform_broadcaster.sendTransform(aruco_frame_wrt_camera_frame)
                # - pub constant transformation frame
                self._pub_tf(time_now=time_now) # TODO: should be in the stage_action(), and use tf locally
                time.sleep(0.5)
                # self._transform_broadcaster.sendTransform(aruco_frame_wrt_camera_frame_before)
                # self._transform_broadcaster.sendTransform(aruco_frame_wrt_camera_frame_after)

                ### TF SOLVER ###
                # - transform:
                try:
                    # zed_T_world
                    camera_to_world = self._tf_buffer.lookup_transform(self._WAM_JOINT_IDs["base_link_frame"], self._WAM_JOINT_IDs["camera_link_frame"], rospy.Time(0))
                    # object_T_world
                    object_to_world_before = tf2_geometry_msgs.do_transform_pose(aruco_frame_wrt_camera_frame_before_pose, camera_to_world)
                    object_to_world_after = tf2_geometry_msgs.do_transform_pose(aruco_frame_wrt_camera_frame_after_pose, camera_to_world)
                    self._print(object_to_world_before)

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e: 
                    rospy.logerr(self._format(
                        info="\n\t[{}] \n\t\tCan't find BEFORE/AFTER transform from aruco to world coord.".format(e)))
                    success = False
                
                ### IK SOLVER ###
                # - solve before:
                if success:
                    wam_joint_states = self._ik_solver.get_ik(
                        self._IK_SEED_STATE_IC,
                        #########################
                        object_to_world_before.pose.position.x,
                        object_to_world_before.pose.position.y,
                        object_to_world_before.pose.position.z,
                        -object_to_world_before.pose.orientation.x, # mirror z 
                        -object_to_world_before.pose.orientation.y, # mirror z 
                        object_to_world_before.pose.orientation.z,
                        object_to_world_before.pose.orientation.w,
                        # -1/np.sqrt(2), 0, 0, 1/np.sqrt(2)
                        #########################
                    )
                    wam_joint_states = np.array(wam_joint_states, dtype=float).reshape(-1).tolist()
                    if len(wam_joint_states) == 1:
                        rospy.logerr(self._format(info="IK solver failed for (BEFORE) ArUco Marker !!"))
                        success = False
                    else:
                        # - cached towards the next stage
                        self._target_aruco_pose_before = wam_joint_states
                # - solve after:
                if success:
                    wam_joint_states = self._ik_solver.get_ik(
                        self._IK_SEED_STATE_IC,
                        #########################
                        object_to_world_after.pose.position.x,
                        object_to_world_after.pose.position.y,
                        object_to_world_after.pose.position.z,
                        -object_to_world_after.pose.orientation.x,
                        -object_to_world_after.pose.orientation.y,
                        object_to_world_after.pose.orientation.z,
                        object_to_world_after.pose.orientation.w,
                        # -1/np.sqrt(2), 0, 0, 1/np.sqrt(2)
                        #########################
                    )
                    wam_joint_states = np.array(wam_joint_states, dtype=float).reshape(-1).tolist()
                    if len(wam_joint_states) == 1:
                        rospy.logerr(self._format(info="IK solver failed for (AFTER) ArUco Marker !!"))
                        success = False
                    else:
                        # - cached towards the next stage
                        self._target_aruco_pose_after = wam_joint_states
                
                ### CMD for this session ###
                if self._target_aruco_pose_before and success:
                    rospy.logwarn(self._format(info="IK solver COMPLETED !! ARM in MOTION ..."))
                    self._target_wam_joints = (self._target_aruco_pose_before)
                #######################  END  #######################
            else:
                success = False
        
        ###################################
        # - Reached ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.IMPROVISATION:
            # -> Invoke to press AruCo Marker
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION:
                ####################### BEGIN #######################
                ### CMD for this session ###
                if self._target_aruco_pose_after:
                    self._target_wam_joints = (self._target_aruco_pose_after)
                #######################  END  #######################            
            else:
                success = False
        
        ###################################
        # - Pressed ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.POST_IMPROVISATION:
            # -> Invoke to release the button
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.RE_IMPROVISATION:
                ####################### BEGIN #######################
                ### CMD for this session ###
                if self._target_aruco_pose_before:
                    self._target_wam_joints = (self._target_aruco_pose_before)
                #######################  END  #######################
            else:
                success = False
        
        ###################################
        # - Released ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.RE_IMPROVISATION:
            # -> Invoke to home, end of the choreography
            if      new_stage is DEMO_STAGE_CHOREOGRAPHY.WINDING:
                ####################### BEGIN #######################
                pass # do nothing, let the winding stage handling the homing
                #######################  END  #######################
            else:
                success = False
        
        ###################################
        # - Unknown:
        else:
            success = False

        ### Update ###
        if success:
            # - perform the actual action:
            self._send_barrett_to_joint_positions_non_block(joint_positions=self._target_wam_joints)
            # - enter the new stage
            self._curr_stage = new_stage
        else:
            # - stuck in stage transition
            rospy.logerr(self._format(info="Unsucessful Stage Transition From [{}] -x-> [{}]!".format(self._curr_stage, new_stage)))
        
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
        
        # - Releasing ArUco Marker:
        elif    self._curr_stage is DEMO_STAGE_CHOREOGRAPHY.RE_IMPROVISATION:
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
    # Helper Static Functions
    #######################
    @staticmethod
    def _array_to_vector3(array):
        return Vector3(array[0], array[1], array[2])
    
    @staticmethod
    def _array_to_quaternion(array):
        return Quaternion(array[0], array[1], array[2], array[3])

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

    def _pub_tf(self, time_now=None):
        #######################
        # ZED CENTER
        #######################
        if time_now is None:
            self._CONST_POSES_IN_WORLD_COORD["zed_link_frame"].header.stamp = rospy.Time.now()
        else:
            self._CONST_POSES_IN_WORLD_COORD["zed_link_frame"].header.stamp = time_now
        self._transform_broadcaster.sendTransform(self._CONST_POSES_IN_WORLD_COORD["zed_link_frame"])

        #######################
        # CAMERA FRAME FOR OBJECT TRANSFORMS
        #######################
        if time_now is None:
            self._CONST_POSES_IN_WORLD_COORD["camera_link_frame"].header.stamp = rospy.Time.now()
        else:
            self._CONST_POSES_IN_WORLD_COORD["camera_link_frame"].header.stamp = time_now
        self._transform_broadcaster.sendTransform(self._CONST_POSES_IN_WORLD_COORD["camera_link_frame"])

    #######################
    # Callback Functions
    #######################
    def _callback_upon_wam_joint_states(self, wam_joint_states_msg):
        if len(wam_joint_states_msg.name) == 7:
            ### Pre-process ###
            time_stamp = wam_joint_states_msg.header.stamp
            pos = wam_joint_states_msg.position
            vel = wam_joint_states_msg.velocity
            eff = wam_joint_states_msg.effort
            ### Capture ###
            with self._wam_joint_state_lock:
                self._wam_joint_position = pos
                self._wam_time_stamp = time_stamp

    def _callback_upon_summit_cmd(self, is_summit_in_position_msg):
        if is_summit_in_position_msg.data:
            ### Pre-process ###
            wam_request = WAM_REQUEST.FAILED
            try: 
                wam_request = WAM_REQUEST(is_summit_in_position_msg.data)
            except ValueError: # treat out-of-scope as failure
                wam_request = WAM_REQUEST.FAILED
                rospy.logerr("Invalid Request [SUMMIT:{}]!".format(is_summit_in_position_msg.data))
            
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
            self._zed_framestamp   = object_in_camera_frame_msg.header.stamp
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
