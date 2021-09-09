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

from gazebo_msgs.msg import ModelState, LinkStates, ContactsState
from gazebo_msgs.srv import GetModelState, SetModelState, GetLinkState, SetLinkState
from gazebo_msgs.srv import SetModelConfiguration, SetModelConfigurationRequest

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Header
from std_srvs.srv import Empty

import std_msgs.msg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2

from trac_ik_python.trac_ik import IK

#######################
#######################

class TFPublisher:

    def __init__(self):

        # WAM links / joints
        self.base_link_frame = 'wam/base_link'
        self.forearm_link_frame = 'wam/forearm_link'
        self.zed_camera_center_frame = 'zed_camera_center'
        self.camera_frame = 'camera_frame'

        # Transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transform_broadcaster = tf2_ros.TransformBroadcaster()

        #######################
        #######################

        self.zed_mesh_file = rospy.get_param('zed_mesh_file', None)

        # Measurements of ZED to forearm link in world coords.
        self.offset_to_left_ZED_lens_x_axis = -135 / 1000  # height from zed to forearm link
        self.offset_to_left_ZED_lens_y_axis = -171 / 1000  # distance from zed to forearm link. TODO: this is a rough measurement.
        self.offset_to_left_ZED_lens_z_axis = 65 / 1000  # camera center to left lens.

        #######################
        # Transforms: ZED tf tree
        #######################

        self.zed_camera_center = geometry_msgs.msg.TransformStamped()
        self.zed_camera_center.header.frame_id = self.forearm_link_frame
        self.zed_camera_center.child_frame_id = self.zed_camera_center_frame

        # offset in world coords.
        self.zed_camera_center_in_forearm_link_frame = PoseStamped()
        self.zed_camera_center_in_forearm_link_frame.header.stamp = rospy.Time.now()
        self.zed_camera_center_in_forearm_link_frame.header.frame_id = self.base_link_frame
        self.zed_camera_center_in_forearm_link_frame.pose.position.x = self.offset_to_left_ZED_lens_x_axis
        self.zed_camera_center_in_forearm_link_frame.pose.position.y = self.offset_to_left_ZED_lens_y_axis
        self.zed_camera_center_in_forearm_link_frame.pose.position.z = 0
        self.zed_camera_center_in_forearm_link_frame.pose.orientation.x = 1 / np.sqrt(2)
        self.zed_camera_center_in_forearm_link_frame.pose.orientation.y = 0
        self.zed_camera_center_in_forearm_link_frame.pose.orientation.z = 0
        self.zed_camera_center_in_forearm_link_frame.pose.orientation.w = 1 / np.sqrt(2)

        #######################
        # Transforms: CAMERA FRAME FOR OBJECT TRANSFORMS
        #######################

        self.zed_left_lens = geometry_msgs.msg.TransformStamped()
        self.zed_left_lens.header.frame_id = self.forearm_link_frame  # self.base_link_frame
        self.zed_left_lens.child_frame_id = self.camera_frame

        # offset in world coords.
        self.camera_in_forearm_link_frame = PoseStamped()
        self.camera_in_forearm_link_frame.header.stamp = rospy.Time.now()
        self.camera_in_forearm_link_frame.header.frame_id = self.base_link_frame
        self.camera_in_forearm_link_frame.pose.position.x = self.offset_to_left_ZED_lens_x_axis
        self.camera_in_forearm_link_frame.pose.position.y = self.offset_to_left_ZED_lens_y_axis
        self.camera_in_forearm_link_frame.pose.position.z = self.offset_to_left_ZED_lens_z_axis
        self.camera_in_forearm_link_frame.pose.orientation.x = 0.5  # 0 # 0.5
        self.camera_in_forearm_link_frame.pose.orientation.y = 0.5  # -1 / np.sqrt(2) # 0.5
        self.camera_in_forearm_link_frame.pose.orientation.z = -0.5  # 1 / np.sqrt(2) # -0.5
        self.camera_in_forearm_link_frame.pose.orientation.w = 0.5  # 0 # 0.5

        #######################
        # PLY FILE
        #######################

        self.zed_wrist_mount = self.load_zed_wrist_mount()
        self.pub_zed_wrist_mount = rospy.Publisher('zed_wrist_mount', PointCloud2, queue_size=1)

        #######################
        # JointState Callbacks
        #######################
        self.joint_states_sub = rospy.Subscriber("/wam/joint_states", JointState, self.get_zed_in_world_transform)

    #######################
    #######################

    def get_zed_in_world_transform(self, data):

        pass

        #######################
        # ZED CENTER
        #######################

        self.zed_camera_center.header.stamp = rospy.Time.now()
        self.zed_camera_center.transform.translation.x = self.zed_camera_center_in_forearm_link_frame.pose.position.x
        self.zed_camera_center.transform.translation.y = self.zed_camera_center_in_forearm_link_frame.pose.position.y
        self.zed_camera_center.transform.translation.z = self.zed_camera_center_in_forearm_link_frame.pose.position.z
        self.zed_camera_center.transform.rotation.x = self.zed_camera_center_in_forearm_link_frame.pose.orientation.x
        self.zed_camera_center.transform.rotation.y = self.zed_camera_center_in_forearm_link_frame.pose.orientation.y
        self.zed_camera_center.transform.rotation.z = self.zed_camera_center_in_forearm_link_frame.pose.orientation.z
        self.zed_camera_center.transform.rotation.w = self.zed_camera_center_in_forearm_link_frame.pose.orientation.w

        self.transform_broadcaster.sendTransform(self.zed_camera_center)

        #######################
        # CAMERA FRAME FOR OBJECT TRANSFORMS
        #######################

        self.zed_left_lens.header.stamp = rospy.Time.now()
        self.zed_left_lens.transform.translation.x = self.camera_in_forearm_link_frame.pose.position.x
        self.zed_left_lens.transform.translation.y = self.camera_in_forearm_link_frame.pose.position.y
        self.zed_left_lens.transform.translation.z = self.camera_in_forearm_link_frame.pose.position.z
        self.zed_left_lens.transform.rotation.x = self.camera_in_forearm_link_frame.pose.orientation.x
        self.zed_left_lens.transform.rotation.y = self.camera_in_forearm_link_frame.pose.orientation.y
        self.zed_left_lens.transform.rotation.z = self.camera_in_forearm_link_frame.pose.orientation.z
        self.zed_left_lens.transform.rotation.w = self.camera_in_forearm_link_frame.pose.orientation.w

        self.transform_broadcaster.sendTransform(self.zed_left_lens)

        #######################
        # Pub Camera Mount
        #######################

        translation = np.array([0, 0, 0]).reshape(-1)
        orientation = np.array([0, 0, 0, 1]).reshape(-1)
        orientation = self.modify_zed_orientation(orientation)
        rotation = np.array(R.from_quat(orientation).as_dcm()).reshape(3, 3)

        # pointcloud
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.camera_frame
        _model_points = np.dot(self.zed_wrist_mount, rotation.T) + translation
        model_points = pcl2.create_cloud_xyz32(header, _model_points)
        self.pub_zed_wrist_mount.publish(model_points)

    #######################
    #######################

    def load_zed_wrist_mount(self):

        ###################################
        # ZED WRIST MOUNT PLY
        ###################################

        rospy.loginfo("")
        rospy.loginfo("*** Loading Mesh file ***")
        input_file = open(self.zed_mesh_file)

        zed_wrist_mount = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            zed_wrist_mount.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        zed_wrist_mount = np.array(zed_wrist_mount)
        input_file.close()

        rospy.loginfo("Loaded ZED Wrist Mount")
        rospy.loginfo("")
        return zed_wrist_mount

    #######################
    #######################

    def modify_zed_orientation(self, q):

        rotation_matrix = np.asarray(R.from_quat(q).as_dcm()).reshape(3, 3)

        #######################
        #######################

        # rotate about x-axis
        theta = np.pi/2
        ccw_x_rotation = np.array([[1, 0, 0],
                                   [0, np.cos(theta), -np.sin(theta)],
                                   [0, np.sin(theta), np.cos(theta)],
                                   ])
        rotation_matrix = np.dot(rotation_matrix, ccw_x_rotation)

        #######################
        #######################

        # SciPy returns quaternion as [x, y, z, w]
        quaternion = np.asarray(R.from_dcm(rotation_matrix).as_quat()).reshape(-1)
        return quaternion

    #######################
    #######################

def main():

    rospy.init_node('barrett_tf_publisher', anonymous=True)
    TFPublisher()

    # rate = rospy.Rate(15.0)
    # while not rospy.is_shutdown():
    #     rate.sleep()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ('Shutting down barrett_tf_publisher')

if __name__ == '__main__':
    main()