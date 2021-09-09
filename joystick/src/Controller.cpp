
#include "barrett_wam_arm_msgs/BarretWamArmTwist.h"
#include "joystick/Controller.h"

#include <geometry_msgs/Twist.h>
#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Float64MultiArray.h>

#include <vector>

TeleopTurtle::TeleopTurtle()
    : left_stick_up_down_(1),
      left_stick_left_right_(0),
      LT_(2),
      right_stick_left_right_(3),
      right_stick_up_down_(4),
      RT_(5),
      cross_key_left_right_(6),
      cross_key_up_down_(7),

      A_(0),
      B_(1),
      X_(2),
      Y_(3),
      LB_(4),
      RB_(5),
      start_(6),
      back_(7),
      power_(8),
      button_stick_left_(9),
      button_stick_right_(10)

{
  // axes
  nh_.param("left_stick_vertical", left_stick_up_down_, left_stick_up_down_);
  nh_.param("left_stick_horizontal", left_stick_left_right_, left_stick_left_right_);
  nh_.param("left_T", LT_, LT_);
  nh_.param("right_stick_vertical", right_stick_up_down_, right_stick_up_down_);
  nh_.param("right_stick_horizontal", left_stick_left_right_, left_stick_left_right_);
  nh_.param("right_T", RT_, RT_);
  nh_.param("cross_key_horizontal", cross_key_left_right_, cross_key_left_right_);
  nh_.param("cross_key_vertical", cross_key_up_down_, cross_key_up_down_);

  // buttons
  nh_.param("A", A_, A_);
  nh_.param("B", B_, B_);
  nh_.param("X", X_, X_);
  nh_.param("Y", Y_, Y_);
  nh_.param("LB", LB_, LB_);
  nh_.param("RB", RB_, RB_);
  nh_.param("start", start_, start_);
  nh_.param("back", back_, back_);
  nh_.param("power", power_, power_);
  nh_.param("button_stick_left", button_stick_left_, button_stick_left_);
  nh_.param("button_stick_right", button_stick_right_, button_stick_right_);
  nh_.param("left_stick_vertical_scale_fast", left_stick_up_down_scale_fast_, left_stick_up_down_scale_fast_);
  nh_.param("left_stick_vertical_scale_slow", left_stick_up_down_scale_slow_, left_stick_up_down_scale_slow_);
  nh_.param("left_stick_angular_scale_fast", left_stick_left_right_scale_fast_, left_stick_left_right_scale_fast_);
  nh_.param("left_stick_angular_scale_slow", left_stick_left_right_scale_slow_, left_stick_left_right_scale_slow_);
  nh_.param("right_stick_vertical_scale", right_stick_up_down_scale_, right_stick_up_down_scale_);
  nh_.param("right_stick_angular_scale", right_stick_left_right_scale_, right_stick_left_right_scale_);

  arm_publisher = nh_.advertise<barrett_wam_arm_msgs::BarretWamArmTwist>("/arm_cartesian_controller/twist_arm_cmds", 1);
  claw_publisher = nh_.advertise<geometry_msgs::Twist>("turtle1/cmd_vel", 1);
  camera_publisher = nh_.advertise<geometry_msgs::Twist>("turtle1/cmd_vel", 1);

  joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &TeleopTurtle::joyCallback, this);
}

void TeleopTurtle::joyCallback(const sensor_msgs::Joy::ConstPtr& joy) {
  ROS_INFO_STREAM("Calling joyCallback!");

  // control mode
  bool drive_control_mode = false;
  bool arm_control_mode = true;

  if (joy->buttons[start_]) {
    arm_control_mode = true;
    drive_control_mode = false;
  }
  if (joy->buttons[back_]) {
    drive_control_mode = true;
    arm_control_mode = false;
  }

  // arm mode on
  if (arm_control_mode) {
  }
  barrett_wam_arm_msgs::BarretWamArmTwist twist_arm_cmds;

  double z_axis_ = 0;
  if (joy->buttons[LB_] && joy->buttons[Y_]) {
    z_axis_ = -0.25;  // down
  } 
  else if (joy->buttons[Y_]) {
    z_axis_ = 0.25;  // up
  } 

  double x_axis_ = 0;
  if (joy->buttons[LB_] && joy->buttons[X_]) {
    x_axis_ = -0.25;
  } 
  else if (joy->buttons[X_]) {
    x_axis_ = 0.25;
  }

  double y_axis_ = 0;
  if (joy->buttons[LB_] && joy->buttons[B_]) {
    y_axis_ = -0.25;
  } 
  else if (joy->buttons[B_]) {
    y_axis_ = 0.25;
  }

  double yaw_axis_ = 0;
  if (joy->buttons[LB_] && joy->buttons[A_]) {
    yaw_axis_ = -0.25;
  } 
  else if (joy->buttons[A_]) {
    yaw_axis_ = 0.25;
  }

  twist_arm_cmds.twist.linear.x = x_axis_;
  twist_arm_cmds.twist.linear.y = y_axis_;
  twist_arm_cmds.twist.linear.z = z_axis_;
  twist_arm_cmds.twist.angular.z = yaw_axis_;
  arm_publisher.publish(twist_arm_cmds);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "teleop_turtle");
  TeleopTurtle teleop_turtle;
  ros::spin();
}