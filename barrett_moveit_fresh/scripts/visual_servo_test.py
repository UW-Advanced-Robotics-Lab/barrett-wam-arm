#!/usr/bin/env python3

import rospy
import moveit_commander
import sys
import numpy as np
from barrett_wam_msgs.msg import RTJointVel
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from tf.transformations import quaternion_matrix, quaternion_from_matrix

IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 720
IMAGE_CENTER = np.array([IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2])

class VisualServoingController:
    def __init__(self):
        rospy.init_node('wam_visual_servoing_control', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)

        self.robot = moveit_commander.RobotCommander()
        self.move_group = moveit_commander.MoveGroupCommander("Manipulator")

        self.pub = rospy.Publisher('/wam/jnt_vel_cmd', RTJointVel, queue_size=1)
        self.rate = rospy.Rate(25)  # Hz

        self.segmentation_center = None
        self.is_tracking = False

        rospy.Subscriber('/segmentation/center', Point, self.segmentation_callback)
        rospy.Subscriber('/segmentation/is_tracking', Bool, self.tracking_callback)

        # Controller gain
        self.kp = 0.002  # Proportional gain: pixels to m/s
        self.max_velocity = 0.5  # m/s max Cartesian speed
        self.deadband = 10  # pixels
        self.kp_orientation = 1.0  # Orientation correction gain
        self.orientation_threshold = 0.05  # radians
        self.max_angular_velocity = 0.5  # rad/s

        # Z-height tracking
        self.initial_z = self.move_group.get_current_pose().pose.position.z
        self.kp_z = 1.0  # Gain for Z axis height correction
        self.max_z_velocity = 0.1  # Limit z speed

        # Initial orientation storage for Z-axis rotation lock
        initial_pose = self.move_group.get_current_pose().pose
        quat = [initial_pose.orientation.x, initial_pose.orientation.y, initial_pose.orientation.z, initial_pose.orientation.w]
        rot_matrix = quaternion_matrix(quat)[:3, :3]
        self.initial_x = rot_matrix[:, 0]  # Initial X axis of end-effector

    def segmentation_callback(self, msg):
        self.segmentation_center = np.array([msg.x, msg.y])

    def tracking_callback(self, msg):
        self.is_tracking = msg.data

    def get_jacobian_and_joints(self):
        joint_values = self.move_group.get_current_joint_values()
        jacobian = self.move_group.get_jacobian_matrix(joint_values)
        return np.array(jacobian), np.array(joint_values)

    def compute_joint_velocities(self, jacobian, cartesian_velocity):
        jacobian_pinv = np.linalg.pinv(jacobian)
        joint_velocities = jacobian_pinv.dot(cartesian_velocity)
        return np.clip(joint_velocities, -0.3, 0.3)

    def compute_orientation_correction(self):
        # Desired orientation: Z downward, X axis aligned with initial X to fix yaw
        desired_z = np.array([0, 0, -1])
        desired_x = self.initial_x

        current_pose = self.move_group.get_current_pose().pose
        quat = [current_pose.orientation.x, current_pose.orientation.y,
                current_pose.orientation.z, current_pose.orientation.w]
        rot_matrix = quaternion_matrix(quat)[:3, :3]
        current_z = rot_matrix[:, 2]  # Z-axis of end-effector
        current_x = rot_matrix[:, 0]  # X-axis of end-effector

        # Z-axis correction
        axis_z = np.cross(current_z, desired_z)
        sin_angle_z = np.linalg.norm(axis_z)
        cos_angle_z = np.dot(current_z, desired_z)

        if sin_angle_z >= 1e-6:
            axis_z /= sin_angle_z
        angle_z = np.arctan2(sin_angle_z, cos_angle_z) if sin_angle_z >= 1e-6 else 0.0
        correction_z = np.zeros(3)
        if abs(angle_z) >= self.orientation_threshold:
            correction_z = self.kp_orientation * angle_z * axis_z

        # X-axis correction (to fix yaw drift)
        axis_x = np.cross(current_x, desired_x)
        sin_angle_x = np.linalg.norm(axis_x)
        cos_angle_x = np.dot(current_x, desired_x)

        if sin_angle_x >= 1e-6:
            axis_x /= sin_angle_x
        angle_x = np.arctan2(sin_angle_x, cos_angle_x) if sin_angle_x >= 1e-6 else 0.0
        correction_x = np.zeros(3)
        if abs(angle_x) >= self.orientation_threshold:
            correction_x = self.kp_orientation * angle_x * axis_x

        total_correction = correction_z + correction_x
        return np.clip(total_correction, -self.max_angular_velocity, self.max_angular_velocity)

    def run(self):
        rospy.loginfo("Visual servoing control loop started...")
        while not rospy.is_shutdown():
            if self.segmentation_center is None:
                rospy.logwarn_throttle(5.0, "Waiting for segmentation center...")
                self.rate.sleep()
                continue

            if not self.is_tracking:
                msg = RTJointVel()
                msg.velocities = [0.0] * 7  # Assuming 7-DOF WAM
                self.pub.publish(msg)
                self.rate.sleep()
                continue

            # Compute error in pixel coordinates (image frame)
            error_pixels = IMAGE_CENTER - self.segmentation_center

            # Apply deadband: zero velocity if error in either direction is below threshold
            velocity_xy = np.zeros(2)
            for i in range(2):
                if abs(error_pixels[i]) >= self.deadband:
                    velocity_xy[i] = self.kp * error_pixels[i]

            velocity_xy = np.clip(velocity_xy, -self.max_velocity, self.max_velocity)

            # Maintain initial Z height
            current_z = self.move_group.get_current_pose().pose.position.z
            error_z = self.initial_z - current_z
            velocity_z = self.kp_z * error_z
            velocity_z = np.clip(velocity_z, -self.max_z_velocity, self.max_z_velocity)

            # Add orientation correction (camera facing downward, fixed yaw)
            angular_velocity = self.compute_orientation_correction()

            cartesian_velocity = np.array([
                -velocity_xy[0], velocity_xy[1], velocity_z,  # vx, vy, vz
                angular_velocity[0], angular_velocity[1], angular_velocity[2]  # wx, wy, wz
            ])

            jacobian, _ = self.get_jacobian_and_joints()
            joint_velocities = self.compute_joint_velocities(jacobian, cartesian_velocity)

            msg = RTJointVel()
            msg.velocities = joint_velocities.tolist()
            rospy.loginfo(f"Seg center: {self.segmentation_center}, Joint vel: {np.round(joint_velocities, 4)}")
            self.pub.publish(msg)

            self.rate.sleep()


if __name__ == '__main__':
    try:
        controller = VisualServoingController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
