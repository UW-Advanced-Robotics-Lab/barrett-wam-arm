#! /usr/bin/env python
from __future__ import division

import sys
import copy
import time

import numpy as np
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

        #######################
        # Object Pose
        #######################

        # self.object_pose_sub = rospy.Subscriber("/aff_densefusion_ros/aff_densefusion_pose", PoseStamped, self.object_pose_callback)
        self.object_pose_sub = rospy.Subscriber("/aruco_single/pose", PoseStamped, self.object_pose_callback)

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
        joint_goal[1] = 0.75
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

        while True:
            self.send_moveit_command()

        #######################
        #######################

        # Trac IK
        # self.pub_ik = rospy.Publisher('/trak_ik/ee_cartesian_pose', PoseStamped, queue_size=1)
        # self.ik_solver = IK(self.base_link_frame, self.ee_link_frame)
        # self.seed_state = [0.01] * self.ik_solver.number_of_joints

    #######################
    #######################

    def object_pose_callback(self, object_in_world_frame_msg):

        #######################
        # transform data
        #######################

        x = object_in_world_frame_msg.pose.position.x
        y = object_in_world_frame_msg.pose.position.y
        z = object_in_world_frame_msg.pose.position.z
        position = np.array([x, y, z])

        x = object_in_world_frame_msg.pose.orientation.x
        y = object_in_world_frame_msg.pose.orientation.y
        z = object_in_world_frame_msg.pose.orientation.z
        w = object_in_world_frame_msg.pose.orientation.w
        orientation = np.array([x, y, z, w])

        # self.wam_observation['Object']['position'] = position
        # self.wam_observation['Object']['orientation'] = orientation

        #######################
        #######################

        # self.compute_ik(position, orientation)
        self.send_moveit_command()

    #######################
    #######################

    def send_moveit_command(self):
        rospy.loginfo('')
        rospy.loginfo("*** Enter '1':ARL or '2':CartesianPath ***")
        input = raw_input()

        #######################
        # ARL
        #######################
        if input == '1':
            # pose_target = Pose()
            # pose_target.position.x = 0.5
            # pose_target.position.y = 0.0
            # pose_target.position.z = 0.75
            # pose_target.orientation.x = 0
            # pose_target.orientation.y = 1
            # pose_target.orientation.z = 0
            # pose_target.orientation.w = 0
            # self.group.set_pose_target(pose_target)
            #
            # self.group.set_pose_target(pose_target)
            # # self.group.plan()
            #
            # # planning: we are just planning, not asking move_group to actually move the robot
            # plan = self.group.plan()
            #
            # # moving: we call the planner to compute the plan and execute it.
            # plan = self.group.go(wait=True)
            # # Calling `stop()` ensures that there is no residual movement
            # self.group.stop()
            # # It is always good to clear your targets after planning with poses.
            # self.group.clear_pose_targets()

            #######################
            #######################

            joint_goal = self.group.get_current_joint_values()
            joint_goal[0] = 0
            joint_goal[1] = 0.95
            joint_goal[2] = 0
            joint_goal[3] = np.pi / 2
            joint_goal[4] = 0
            joint_goal[5] = 0
            joint_goal[6] = 0

            self.group.go(joint_goal, wait=True)
            self.group.stop()

        #######################
        # CartesianPath
        #######################
        elif input == '2':
            waypoints = []

            # wpose = self.group.get_current_pose().pose
            # wpose.position.z -= scale * 0.1  # First move up (z)
            # wpose.position.y += scale * 0.2  # and sideways (y)
            # waypoints.append(copy.deepcopy(wpose))
            #
            # wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
            # waypoints.append(copy.deepcopy(wpose))
            #
            # wpose.position.y -= scale * 0.1  # Third move sideways (y)
            # waypoints.append(copy.deepcopy(wpose))

            wpose = self.group.get_current_pose().pose

            for _ in range(3):
                # 1st
                _wpose = copy.deepcopy(wpose)
                _wpose.position.z -= 0.25
                _wpose.position.y += 0.75
                waypoints.append(_wpose)

                # 2nd: RESET !!!
                waypoints.append(wpose)

                # 3rd
                _wpose = copy.deepcopy(wpose)
                _wpose.position.z -= 0.25
                _wpose.position.y -= 0.75
                waypoints.append(_wpose)

                # 4th: RESET !!!
                waypoints.append(wpose)

            plan, fraction = self.group.compute_cartesian_path(waypoints, # waypoints to follow
                                                               0.01,      # eef_step
                                                               0.0)       # jump_threshold

            # moving: we call the planner to compute the plan and execute it.
            self.group.execute(plan, wait=True)
            # Calling `stop()` ensures that there is no residual movement
            self.group.stop()

        else:
            print("***** Did not enter valid config! *****")

        # time.sleep(10)

    #######################
    #######################

    # def compute_ik(self, position, orientation):
    #
    #     print("")
    #     print("[object-pose] position: ", position)
    #     print("[object-pose] orientation: ", orientation)
    #
    #     # wam_joint_states = self.ik_solver.get_ik(self.seed_state,
    #     #                                                position[0], position[1], position[2],
    #     #                                          orientation[0], orientation[1], orientation[2], orientation[3],
    #     #                                         )
    #
    #     wam_joint_states = self.ik_solver.get_ik(self.seed_state,
    #                                              0.8, 0.05, 0.35, # X, Y, Z
    #                                              0, 1, 0, 0     # Quaternion
    #                                              # orientation[0], orientation[1], orientation[2], orientation[3],  # Quaternion
    #                                              )
    #
    #     print("[trac-ik]     /wam/joint_states: ", wam_joint_states)

def main():

    rospy.init_node('control_barrett_wam_arm', anonymous=True)
    ControlWAMArm()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main()