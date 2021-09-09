#! /usr/bin/env python
from __future__ import division

import sys
import copy

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

class TFPublisher():

    def __init__(self):

        # WAM links / joints
        self.base_link_frame = 'wam/base_link'
        self.forearm_link_frame = 'wam/forearm_link'
        self.ee_link_frame = 'wam/wrist_palm_stump_link' # 'wam/bhand/bhand_grasp_link'
        self.zed_link_frame = 'zed_camera_center'        # 'zed_camera_center' or 'camera_frame'
        self.camera_link_frame = 'camera_frame'
        self.object_frame = 'object_frame'

        # JointState Callbacks
        self.joint_states_sub = rospy.Subscriber("joint_states", JointState, self.get_zed_in_world_transform)

        # Transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transform_broadcaster = tf2_ros.TransformBroadcaster()

        #######################
        # ZED Measurements
        #######################

        self.offset_to_left_ZED_lens_x_axis = -87.5 / 1000 # zed height
        self.offset_to_left_ZED_lens_y_axis = -150 / 1000  # TODO: GET A MEASUREMENT !!!
        self.offset_to_left_ZED_lens_z_axis = 60 / 1000    # left lens

        #######################
        # Transforms: ZED tf tree
        #######################

        self.zed_frame = geometry_msgs.msg.TransformStamped()
        self.zed_frame.header.frame_id = self.forearm_link_frame
        self.zed_frame.child_frame_id = self.zed_link_frame

        self.zed_in_forearm_frame = PoseStamped()
        self.zed_in_forearm_frame.header.stamp = rospy.Time.now()
        self.zed_in_forearm_frame.header.frame_id = self.base_link_frame
        self.zed_in_forearm_frame.pose.position.x = self.offset_to_left_ZED_lens_x_axis
        self.zed_in_forearm_frame.pose.position.y = self.offset_to_left_ZED_lens_y_axis
        self.zed_in_forearm_frame.pose.position.z = 0 # self.offset_to_left_ZED_lens_z_axis
        self.zed_in_forearm_frame.pose.orientation.x = 1 / np.sqrt(2)
        self.zed_in_forearm_frame.pose.orientation.y = 0
        self.zed_in_forearm_frame.pose.orientation.z = 0
        self.zed_in_forearm_frame.pose.orientation.w = 1 / np.sqrt(2)

        #######################
        # Transforms: DUMMY CAMERA LINK FOR OBJECT TRANSFORMS
        #######################

        self.camera_frame = geometry_msgs.msg.TransformStamped()
        self.camera_frame.header.frame_id = self.forearm_link_frame # self.base_link_frame
        self.camera_frame.child_frame_id = self.camera_link_frame

        self.camera_in_forearm_frame = PoseStamped()
        self.camera_in_forearm_frame.header.stamp = rospy.Time.now()
        self.camera_in_forearm_frame.header.frame_id = self.base_link_frame  # self.camera_frame or self.base_frame
        self.camera_in_forearm_frame.pose.position.x = self.offset_to_left_ZED_lens_x_axis
        self.camera_in_forearm_frame.pose.position.y = self.offset_to_left_ZED_lens_y_axis
        self.camera_in_forearm_frame.pose.position.z = self.offset_to_left_ZED_lens_z_axis
        self.camera_in_forearm_frame.pose.orientation.x = 0.5  # 1 / np.sqrt(2)
        self.camera_in_forearm_frame.pose.orientation.y = 0.5  # 0
        self.camera_in_forearm_frame.pose.orientation.z = -0.5 # 0
        self.camera_in_forearm_frame.pose.orientation.w = 0.5  # 1 / np.sqrt(2)

        #######################
        # Transforms: Object Pose
        #######################

        # self.object_pose_sub = rospy.Subscriber("/aff_densefusion_ros/aff_densefusion_pose", PoseStamped, self.object_pose_callback)
        self.object_pose_sub = rospy.Subscriber("/aruco_single/pose", PoseStamped, self.object_pose_callback)
        self.object_pose_pub = rospy.Publisher('object_pose', Marker, queue_size=10)

        self.object_in_world_frame = geometry_msgs.msg.TransformStamped()
        self.object_in_world_frame.header.frame_id = self.base_link_frame
        self.object_in_world_frame.child_frame_id = self.object_frame

        #######################
        #######################

        # Trac IK
        self.pub_ik = rospy.Publisher('/trak_ik/ee_cartesian_pose', PoseStamped, queue_size=1)
        self.ik_solver = IK(self.base_link_frame, self.ee_link_frame)
        self.seed_state = [0.01] * self.ik_solver.number_of_joints

        # MOVEIT
        # self.pub_trajectory = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=1)
        #
        # moveit_commander.roscpp_initialize(sys.argv)
        # self.robot = moveit_commander.RobotCommander()
        # self.scene = moveit_commander.PlanningSceneInterface()
        # self.group = moveit_commander.MoveGroupCommander("barrett_wam_arm")

    #######################
    #######################

    def get_zed_in_world_transform(self, data):
        ''' zed_T_world = zed_T_forearm * forearm_T_world '''

        #######################
        # ZED CENTER
        #######################

        self.zed_frame.header.stamp = rospy.Time.now()
        self.zed_frame.transform.translation.x = self.zed_in_forearm_frame.pose.position.x
        self.zed_frame.transform.translation.y = self.zed_in_forearm_frame.pose.position.y
        self.zed_frame.transform.translation.z = self.zed_in_forearm_frame.pose.position.z
        self.zed_frame.transform.rotation.x = self.zed_in_forearm_frame.pose.orientation.x
        self.zed_frame.transform.rotation.y = self.zed_in_forearm_frame.pose.orientation.y
        self.zed_frame.transform.rotation.z = self.zed_in_forearm_frame.pose.orientation.z
        self.zed_frame.transform.rotation.w = self.zed_in_forearm_frame.pose.orientation.w

        self.transform_broadcaster.sendTransform(self.zed_frame)

        #######################
        # DUMMY CAMERA LINK FOR OBJECT TRANSFORMS
        #######################

        self.camera_frame.header.stamp = rospy.Time.now()
        self.camera_frame.transform.translation.x = self.camera_in_forearm_frame.pose.position.x
        self.camera_frame.transform.translation.y = self.camera_in_forearm_frame.pose.position.y
        self.camera_frame.transform.translation.z = self.camera_in_forearm_frame.pose.position.z
        self.camera_frame.transform.rotation.x = self.camera_in_forearm_frame.pose.orientation.x
        self.camera_frame.transform.rotation.y = self.camera_in_forearm_frame.pose.orientation.y
        self.camera_frame.transform.rotation.z = self.camera_in_forearm_frame.pose.orientation.z
        self.camera_frame.transform.rotation.w = self.camera_in_forearm_frame.pose.orientation.w

        self.transform_broadcaster.sendTransform(self.camera_frame)

        #######################
        #######################

        # self.send_moveit_command()

    #######################
    #######################

    def object_pose_callback(self, object_in_camera_frame_msg):

        # listener = tf.TransformListener()
        # listener.waitForTransform(self.base_link_frame, self.forearm_link_frame, rospy.Time(), rospy.Duration(1.0))
        # todo: cantransfrom()

        #######################
        # transform data
        #######################
        ''' object_T_world = object_T_zed * zed_T_world '''

        # zed_T_world                                        target frame             source frame
        camera_to_world = self.tf_buffer.lookup_transform(self.base_link_frame, self.camera_link_frame, rospy.Time(0))
        # object_T_world
        object_to_world = tf2_geometry_msgs.do_transform_pose(object_in_camera_frame_msg, camera_to_world)

        x = object_to_world.pose.position.x
        y = object_to_world.pose.position.y
        z = object_to_world.pose.position.z
        position = np.array([x, y, z])

        x = object_to_world.pose.orientation.x
        y = object_to_world.pose.orientation.y
        z = object_to_world.pose.orientation.z
        w = object_to_world.pose.orientation.w
        orientation = np.array([x, y, z, w])

        # self.wam_observation['Object']['position'] = position
        # self.wam_observation['Object']['orientation'] = orientation

        #######################
        # publish transform
        #######################

        self.object_in_world_frame.header.stamp = rospy.Time.now()
        self.object_in_world_frame.transform.translation.x = position[0]
        self.object_in_world_frame.transform.translation.y = position[1]
        self.object_in_world_frame.transform.translation.z = position[2]
        self.object_in_world_frame.transform.rotation.x = orientation[0]
        self.object_in_world_frame.transform.rotation.y = orientation[1]
        self.object_in_world_frame.transform.rotation.z = orientation[2]
        self.object_in_world_frame.transform.rotation.w = orientation[3]

        self.transform_broadcaster.sendTransform(self.object_in_world_frame)

        #######################
        #######################

        self.compute_ik(position, orientation)
        # self.send_moveit_command()

    #######################
    #######################

    def compute_ik(self, position, orientation):

        print("")
        print("[object-pose] position: ", position)
        print("[object-pose] orientation: ", orientation)

        # wam_joint_states = self.ik_solver.get_ik(self.seed_state,
        #                                                position[0], position[1], position[2],
        #                                          orientation[0], orientation[1], orientation[2], orientation[3],
        #                                         )

        wam_joint_states = self.ik_solver.get_ik(self.seed_state,
                                                 0.8, 0.05, 0.35, # X, Y, Z
                                                 0, 1, 0, 0     # Quaternion
                                                 # orientation[0], orientation[1], orientation[2], orientation[3],  # Quaternion
                                                 )

        print("[trac-ik]     /wam/joint_states: ", wam_joint_states)

    #######################
    #######################

    # def send_moveit_command(self):
    #     print("***** Enter 'Home' or 'CollectData' or 'ObjectPose' *****")
    #     input = raw_input()
    #     if input == 'Home':
    #         ###############
    #         pose_target = Pose()
    #         pose_target.position.x = 0.25
    #         pose_target.position.y = 0.0
    #         pose_target.position.z = 0.25
    #         pose_target.orientation.x = 0
    #         pose_target.orientation.y = 0
    #         pose_target.orientation.z = 1
    #         pose_target.orientation.w = 0
    #
    #         self.group.set_pose_target(pose_target)
    #         # self.group.plan()
    #
    #         ## Now, we call the planner to compute the plan and execute it.
    #         plan = self.group.go(wait=False)
    #
    #         # Calling `stop()` ensures that there is no residual movement
    #         self.group.stop()
    #
    #         # It is always good to clear your targets after planning with poses.
    #         # Note: there is no equivalent function for clear_joint_value_targets()
    #         self.group.clear_pose_targets()
    #
    #     elif input == 'CollectData':
    #         pose_target = Pose()
    #         pose_target.position.x = 0.75
    #         pose_target.position.y = 0.0
    #         pose_target.position.z = 0.5
    #         pose_target.orientation.x = 0
    #         pose_target.orientation.y = 1
    #         pose_target.orientation.z = 0
    #         pose_target.orientation.w = 0
    #
    #         self.group.set_pose_target(pose_target)
    #         # self.group.plan()
    #
    #         ## Now, we call the planner to compute the plan and execute it.
    #         plan = self.group.go(wait=True)
    #
    #         # Calling `stop()` ensures that there is no residual movement
    #         self.group.stop()
    #
    #         # It is always good to clear your targets after planning with poses.
    #         # Note: there is no equivalent function for clear_joint_value_targets()
    #         self.group.clear_pose_targets()
    #
    #     elif input == 'ObjectPose':
    #         pose_target = Pose()
    #         pose_target.position.x = self.wam_observation['Object']['position'][0]
    #         pose_target.position.y = self.wam_observation['Object']['position'][1]
    #         pose_target.position.z = self.wam_observation['Object']['position'][2]
    #         pose_target.orientation.x = self.wam_observation['Object']['orientation'][0]
    #         pose_target.orientation.y = self.wam_observation['Object']['orientation'][1]
    #         pose_target.orientation.z = self.wam_observation['Object']['orientation'][2]
    #         pose_target.orientation.w = self.wam_observation['Object']['orientation'][3]
    #
    #         self.group.set_pose_target(pose_target)
    #         # self.group.plan()
    #
    #         ## Now, we call the planner to compute the plan and execute it.
    #         plan = self.group.go(wait=False)
    #
    #         # Calling `stop()` ensures that there is no residual movement
    #         self.group.stop()
    #
    #         # It is always good to clear your targets after planning with poses.
    #         # Note: there is no equivalent function for clear_joint_value_targets()
    #         self.group.clear_pose_targets()
    #
    #     else:
    #         print("***** Did not enter valid config! *****")

def main():

    rospy.init_node('barrett_tf_publisher', anonymous=True)
    TFPublisher()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        rate.sleep()
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print ('Shutting down Moveit')

if __name__ == '__main__':
    main()