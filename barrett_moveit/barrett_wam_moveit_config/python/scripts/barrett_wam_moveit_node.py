#! /usr/bin/env python
from __future__ import division

import sys
import copy
import time

import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R

import rospy

import tf
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg

from gazebo_msgs.msg import ModelState, LinkState, ContactsState
from gazebo_msgs.srv import GetModelState, SetModelState, GetLinkState, SetLinkState
from gazebo_msgs.srv import SetModelConfiguration, SetModelConfigurationRequest

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Header
from std_srvs.srv import Empty

from visualization_msgs.msg import Marker

from trac_ik_python.trac_ik import IK

import moveit_commander
import moveit_msgs.msg

#######################
#######################

class ControlWAMArm():

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
        self.ee_link_frame = 'wam/bhand/bhand_grasp_link'  # 'wam/bhand/bhand_grasp_link'
        self.zed_link_frame = 'zed_camera_center'  # 'zed_camera_center' or 'camera_frame'
        self.camera_link_frame = 'camera_frame'
        self.object_frame = 'object_frame'
        self.grasp_frame = 'grasp_frame'

        #######################
        # MOVEIT
        #######################

        self.pub_trajectory = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=1)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("arm")

        rospy.loginfo('')
        rospy.loginfo('*** Moveit Configs ***')
        rospy.loginfo('Valid Groups: {}'.format(self.robot.get_group_names()))
        # print('Current State: {}'.format(self.robot.get_current_state()))
        rospy.loginfo('Reference frame: {}'.format(self.group.get_planning_frame()))
        rospy.loginfo('End Effector frame: {}'.format(self.group.get_end_effector_link()))

        #######################
        #######################

        # We get the joint values from the group and change some of the values:
        joint_goal = self.group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = 0.5
        joint_goal[2] = 0
        joint_goal[3] = np.pi / 2
        joint_goal[4] = 0
        joint_goal[5] = 0
        joint_goal[6] = 0

        self.group.go(joint_goal, wait=True)
        self.group.stop()

        # time.sleep(10)

        #######################
        #######################

        # Trac IK
        # self.pub_ik = rospy.Publisher('/trak_ik/ee_cartesian_pose', PoseStamped, queue_size=1)
        # self.ik_solver = IK(self.base_link_frame, self.ee_link_frame)
        # self.seed_state = [0.01] * self.ik_solver.number_of_joints

        #######################
        # Object Pose
        #######################

        self.object_pose_sub = rospy.Subscriber("/arl_vicon_ros/aff_densefusion_pose", PoseStamped, self.object_pose_callback)
        # self.object_pose_sub = rospy.Subscriber("/aruco_single/pose", PoseStamped, self.object_pose_callback)

    #######################
    #######################

    def object_pose_callback(self, object_in_world_frame_msg):
        #######################
        #######################

        x = object_in_world_frame_msg.pose.position.x
        y = object_in_world_frame_msg.pose.position.y
        z = object_in_world_frame_msg.pose.position.z
        position = np.array([x, y, z])

        w = object_in_world_frame_msg.pose.orientation.w
        x = object_in_world_frame_msg.pose.orientation.x
        y = object_in_world_frame_msg.pose.orientation.y
        z = object_in_world_frame_msg.pose.orientation.z
        orientation = np.array([x, y, z, w])

        #######################
        # modify for gripper
        #######################

        position = np.array([0.75, 0, 0.25])
        # position[2] = position[2] + 5 / 100  # offset for: 'wam/bhand/bhand_grasp_link'
        position[2] = position[2] + 15/100 # offset for: 'wam/wrist_palm_link'

        orientation = self.modify_obj_rotation_matrix_for_grasping(q=orientation)

        print('')
        self.print_object_pose(q=orientation, t=position)

        #######################
        # pub tf
        #######################

        # tf
        grasp_frame_on_object = geometry_msgs.msg.TransformStamped()
        grasp_frame_on_object.header.frame_id = self.base_link_frame
        grasp_frame_on_object.child_frame_id = self.grasp_frame
        grasp_frame_on_object.header.stamp = rospy.Time.now()
        grasp_frame_on_object.transform.translation.x = position[0]
        grasp_frame_on_object.transform.translation.y = position[1]
        grasp_frame_on_object.transform.translation.z = position[2]
        grasp_frame_on_object.transform.rotation.x = orientation[0]
        grasp_frame_on_object.transform.rotation.y = orientation[1]
        grasp_frame_on_object.transform.rotation.z = orientation[2]
        grasp_frame_on_object.transform.rotation.w = orientation[3]
        self.transform_broadcaster.sendTransform(grasp_frame_on_object)

        #######################
        # MoveIt!
        #######################

        self.send_moveit_command(position, orientation)

    #######################
    #######################

    def send_moveit_command(self, position, orientation):
        rospy.loginfo('')
        rospy.loginfo("*** Enter '1':ARL or '2':DenseFusion ***")
        input = raw_input()

        #######################
        # ARL
        #######################
        if input == '1':
            # We get the joint values from the group and change some of the values:
            joint_goal = self.group.get_current_joint_values()
            joint_goal[0] = 0
            joint_goal[1] = 0.5
            joint_goal[2] = 0
            joint_goal[3] = np.pi / 2
            joint_goal[4] = 0
            joint_goal[5] = 0
            joint_goal[6] = 0

            self.group.go(joint_goal, wait=True)
            self.group.stop()

        #######################
        # DenseFusion
        #######################
        elif input == '2':
            pose_target = Pose()
            pose_target.position.x = position[0]
            pose_target.position.y = position[1]
            pose_target.position.z = position[2]
            pose_target.orientation.x = orientation[0]
            pose_target.orientation.y = orientation[1]
            pose_target.orientation.z = orientation[2]
            pose_target.orientation.w = orientation[3]
            self.group.set_pose_target(pose_target)

            self.group.set_pose_target(pose_target)
            # self.group.plan()

            # planning: we are just planning, not asking move_group to actually move the robot
            self.group.plan()

            # moving: we call the planner to compute the plan and execute it.
            self.group.go(wait=True)
            # Calling `stop()` ensures that there is no residual movement
            self.group.stop()
            # It is always good to clear your targets after planning with poses.
            self.group.clear_pose_targets()

        else:
            print("***** Did not enter valid config! *****")

    #######################
    #######################

    def print_object_pose(self, q, t):
        # convert translation to [cm]
        t = t.copy() * 100

        rot = np.array(R.from_quat(q).as_dcm()).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(rot)
        rvec = rvec * 180 / np.pi
        rvec = np.squeeze(np.array(rvec)).reshape(-1)

        rospy.loginfo('position    [cm]:  x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(t[0], t[1], t[2]))
        rospy.loginfo('orientation [deg]: x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(rvec[0], rvec[1], rvec[2]))

    #######################
    #######################

    def modify_obj_rotation_matrix_for_grasping(self, q):
        #######################
        #######################

        # theta = np.pi/2
        # ccw_x_rotation = np.array([[1, 0, 0],
        #                            [0, np.cos(theta), -np.sin(theta)],
        #                            [0, np.sin(theta), np.cos(theta)],
        #                            ])
        #
        # ccw_y_rotation = np.array([[np.cos(theta), 0 , np.sin(theta)],
        #                            [0, 1, 0],
        #                            [-np.sin(theta), 0, np.cos(theta)],
        #                            ])
        #
        # ccw_z_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
        #                            [np.sin(theta), np.cos(theta), 0],
        #                            [0, 0, 1],
        #                            ])

        #######################
        #######################

        rot = np.asarray(R.from_quat(q).as_dcm()).reshape(3, 3)

        # rotate about z-axis
        theta = np.pi / 2
        ccw_z_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta), np.cos(theta), 0],
                                   [0, 0, 1],
                                   ])
        rot = np.dot(rot, ccw_z_rotation)

        # rotate about x-axis
        theta = np.pi
        ccw_x_rotation = np.array([[1, 0, 0],
                                   [0, np.cos(theta), -np.sin(theta)],
                                   [0, np.sin(theta), np.cos(theta)],
                                   ])
        rot = np.dot(rot, ccw_x_rotation)

        # rotate about y-axis
        theta = np.pi / 2
        ccw_y_rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
                                   [0, 1, 0],
                                   [-np.sin(theta), 0, np.cos(theta)],
                                   ])
        rot = np.dot(rot, ccw_y_rotation)

        #######################
        #######################

        return np.asarray(R.from_dcm(rot).as_quat()).reshape(-1)

    #######################
    #######################

def main():

    rospy.init_node('control_barrett_wam_arm', anonymous=True)
    ControlWAMArm()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main()