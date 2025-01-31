#!/usr/bin/env python3

import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped, Twist
from barrett_wam_msgs.srv import JointMove  # Correct service type import
from std_srvs.srv import Empty  # For the go_home service
import sys
import copy
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import warnings
import time
from actionlib_msgs.msg import GoalStatusArray



# Global variable to store the latest ArUco marker pose
aruco_pose = None
aruco_received = False  # Flag to track when a new ArUco message is received

# Global flag to track if status 2 (Preempted) was received
preempted_received = False  

def goal_status_callback(msg):
    global preempted_received
    
    if not msg.status_list:  # Ensure the list isn't empty
        return
    
    status = msg.status_list[-1]

    
    if status.status == 1:  # Status 2 means "Preempted"
            
        preempted_received = True  # Set flag to allow goF execution

    if status.status == 3 and preempted_received:  # Status 3 means "Goal Reached"
        rospy.loginfo("Goal reached! Executing goF command.")
        send_velocity_command()  # Execute goF command
        preempted_received = False  # Reset flag to wait for the next status 2
            

def aruco_pose_callback(msg):
    """Callback function for the /aruco_single/pose topic."""
    global aruco_pose, aruco_received
    aruco_pose = msg.pose  # Store the received pose
    aruco_received = True  # Set the flag to indicate a new message was received

def call_empty_service(service_name):
    rospy.loginfo(f"Calling {service_name} service...")
    rospy.wait_for_service(service_name)
    try:
        empty_service = rospy.ServiceProxy(service_name, Empty)
        empty_service()
        rospy.loginfo(f"Service {service_name} called successfully.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call to {service_name} failed: {e}")

def move_cartesian_path(x, y, z, qx, qy, qz, qw):
    move_group.set_start_state_to_current_state()
    waypoints = []
    pose_goal = Pose()
    pose_goal.position.x = x
    pose_goal.position.y = y
    pose_goal.position.z = z
    pose_goal.orientation.x = qx
    pose_goal.orientation.y = qy
    pose_goal.orientation.z = qz
    pose_goal.orientation.w = qw
    waypoints.append(pose_goal)
    move_group.set_pose_target(pose_goal)
    (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.01, False)
    rospy.loginfo(f"Fraction: {fraction}")
    if fraction > 0.95:
        rospy.loginfo("Successfully planned the Cartesian path.")
        joint_trajectory = plan.joint_trajectory
        final_joint_positions = joint_trajectory.points[-1].positions
        rospy.loginfo("Calculated final joint positions (without execution):")
        rospy.loginfo(final_joint_positions)
        call_joint_move_service(final_joint_positions)
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    display_trajectory_publisher.publish(display_trajectory)

def call_joint_move_service(joint_positions):
    rospy.loginfo("Calling /wam/joint_move service...")
    rospy.wait_for_service("/wam/joint_move")
    try:
        joint_move_service = rospy.ServiceProxy("/wam/joint_move", JointMove)
        response = joint_move_service(joint_positions)
        rospy.loginfo("Service response: %s", response)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

def move_to_joint_positions(joint_positions):
    rospy.loginfo("Sending joint positions to /wam/joint_move service...")
    call_joint_move_service(joint_positions)

def go_home():
    call_empty_service("/wam/go_home")

def move_to_pick():
    rospy.loginfo("Moving to 'pick' position...")
    pick_positions = [-0.3, 0.6909, 0, 1.3406, 0, 1.1206, -0.3]
    move_to_joint_positions(pick_positions)

def move_to_pick2():
    rospy.loginfo("Moving to 'pick' position...")
    pick_positions = [0.00770642729152072, 0.5621428710295964, 0.004971719719292722, 1.19011342793461, 0.004111701080930619, 1.404462203836328, 0.012329383425738577]
    move_to_joint_positions(pick_positions)


def move_to_place():
    rospy.loginfo("Moving to 'place' position...")
    place_positions = [-0.3, 0.6909, 0, 1.3406, 0, 1.1206, -0.3]
    move_to_joint_positions(place_positions)
    
def move_to_place2():
    rospy.loginfo("Moving to 'place' position...")
    place_positions = [-2.6, 0.90, -0.886, 0.911, 0.702, 1.438, -0.210]
    move_to_joint_positions(place_positions)

def store_bottle():
    place_positions = [0, -1.081, 1.561, 2.9, 1.240, 0.999, 0.26]
    move_to_joint_positions(place_positions)

def wait_for_aruco_pose():
    """Waits for the next ArUco pose to be received and stores it."""
    global aruco_pose, aruco_received
    aruco_received = False  # Reset the flag
    rospy.loginfo("Waiting for the next ArUco marker pose...")
    while not aruco_received and not rospy.is_shutdown():
        rospy.sleep(0.1)  # Sleep to avoid busy-waiting
    return aruco_pose

def find_and_move():
    """Waits for the next ArUco marker pose, calculates a new position, and moves the robot."""
    aruco_marker_pose = wait_for_aruco_pose()
    if aruco_marker_pose is None:
        rospy.logwarn("No ArUco pose received. Aborting find operation.")
        return

    # Get the current robot pose
    current_pose = move_group.get_current_pose().pose
    rospy.loginfo("Current robot pose:")
    rospy.loginfo(f"Position: {current_pose.position}")
    rospy.loginfo(f"Orientation: {current_pose.orientation}")

    # Calculate the new position
    new_x = current_pose.position.x + aruco_marker_pose.position.x + 0.06
    new_y = current_pose.position.y - aruco_marker_pose.position.y + 0.1
    new_z = current_pose.position.z - aruco_marker_pose.position.z + 0.1
    rospy.loginfo(f"Moving to new position: x={new_x}, y={new_y}, z={new_z}")

    # Move to the new position
    move_cartesian_path(
        new_x,
        new_y,
        new_z,
        current_pose.orientation.x,
        current_pose.orientation.y,
        current_pose.orientation.z,
        current_pose.orientation.w,
    )
    
def go_to_pickup():
    
    summit_pub = rospy.Publisher('/uwarl/move_base_simple/goal', PoseStamped, queue_size=10)
        
    rospy.sleep(1)
        
    # Create PoseStamped message
    goal_msg = PoseStamped()
    goal_msg.header.frame_id = "uwarl_map"
    goal_msg.header.stamp = rospy.Time.now()
    
    # Set position
    goal_msg.pose.position.x = 1.279
    goal_msg.pose.position.y = -1.354
    goal_msg.pose.position.z = 0.0  # Assuming 2D movement
    
    # Set orientation (quaternion)
    goal_msg.pose.orientation.x = 0.0
    goal_msg.pose.orientation.y = 0.0
    goal_msg.pose.orientation.z = 0.316
    goal_msg.pose.orientation.w = 0.948
    
    # Publish the message
    rospy.loginfo("Publishing goal: x=1.279, y=-1.354, quat_z=0.361, quat_w=0.9325")
    summit_pub.publish(goal_msg)

def send_velocity_command():
    """Send velocity commands to move forward for 1 second."""
    rospy.loginfo("Sending velocity command to move forward...")
    cmd_vel_pub = rospy.Publisher('/uwarl/move_base/cmd_vel', Twist, queue_size=10)

    # Create the Twist message
    velocity_msg = Twist()
    velocity_msg.linear.x = 0.2  # Move forward at 0.1 m/s
    velocity_msg.linear.y = 0.0
    velocity_msg.linear.z = 0.0
    velocity_msg.angular.x = 0.0
    velocity_msg.angular.y = 0.0
    velocity_msg.angular.z = 0.0

    # Publish the message repeatedly for 1 second
    start_time = time.time()
    while time.time() - start_time < 3.0:
        cmd_vel_pub.publish(velocity_msg)
        rospy.sleep(0.1)  # Publish at 10 Hz

    # Stop the robot after 1 second
    velocity_msg.linear.x = 0.0
    cmd_vel_pub.publish(velocity_msg)
    rospy.loginfo("Velocity command completed.")

def send_velocity_command2():
    """Send velocity commands to move forward for 1 second."""
    rospy.loginfo("Sending velocity command to move forward...")
    cmd_vel_pub = rospy.Publisher('/uwarl/move_base/cmd_vel', Twist, queue_size=10)

    # Create the Twist message
    velocity_msg = Twist()
    velocity_msg.linear.x = -0.2  # Move forward at 0.1 m/s
    velocity_msg.linear.y = 0.0
    velocity_msg.linear.z = 0.0
    velocity_msg.angular.x = 0.0
    velocity_msg.angular.y = 0.0
    velocity_msg.angular.z = -0.2

    # Publish the message repeatedly for 1 second
    start_time = time.time()
    while time.time() - start_time < 3.0:
        cmd_vel_pub.publish(velocity_msg)
        rospy.sleep(0.1)  # Publish at 10 Hz

    # Stop the robot after 1 second
    velocity_msg.linear.x = 0.0
    cmd_vel_pub.publish(velocity_msg)
    rospy.loginfo("Velocity command completed.")

def go_to_return():
    
    summit_pub = rospy.Publisher('/uwarl/move_base_simple/goal', PoseStamped, queue_size=10)
        
    rospy.sleep(1)
        
    # Create PoseStamped message
    goal_msg = PoseStamped()
    goal_msg.header.frame_id = "uwarl_map"
    goal_msg.header.stamp = rospy.Time.now()
    
    # Set position
    goal_msg.pose.position.x = -5.319
    goal_msg.pose.position.y = -11.949
    goal_msg.pose.position.z = 0.0  # Assuming 2D movement
    
    # Set orientation (quaternion)
    goal_msg.pose.orientation.x = 0.0
    goal_msg.pose.orientation.y = 0.0
    goal_msg.pose.orientation.z = -0.9333
    goal_msg.pose.orientation.w = 0.3588
    
    # Publish the message
    rospy.loginfo("Publishing goal: x=1.279, y=-1.354, quat_z=0.361, quat_w=0.9325")
    summit_pub.publish(goal_msg)


if __name__ == "__main__":
    
    rospy.set_param('/rosout/level', 4)  # Set the level to ERROR
    warnings.filterwarnings("ignore")
    
    rospy.Subscriber("/uwarl/move_base/status", GoalStatusArray, goal_status_callback)

    try:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("wam_cartesian_path", anonymous=True)

        # Initialize MoveIt Commander
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        move_group = moveit_commander.MoveGroupCommander("Manipulator")

        move_group.set_pose_reference_frame("wam_link_footprint")
        end_effector_link = move_group.get_end_effector_link()
        rospy.loginfo(f"End effector link: {end_effector_link}")
        move_group.set_goal_tolerance(0.10)
        move_group.set_goal_orientation_tolerance(0.1)
        move_group.set_planning_time(20.0)  # Increase the planning time
        move_group.set_num_planning_attempts(100)

        # Display trajectory publisher
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                        moveit_msgs.msg.DisplayTrajectory,
                                                        queue_size=20)

        # Subscribe to the /aruco_single/pose topic
        rospy.Subscriber("/aruco_single/pose", PoseStamped, aruco_pose_callback)

        while not rospy.is_shutdown():
            user_input = input(
                "Enter 'p' for Cartesian pose, 'j' for joint positions, 'h' to go home, 'pick', 'place', 'aruco', 'find', or 'q' to quit: "
            )

            if user_input.lower() == "q":
                rospy.loginfo("Exiting...")
                break

            try:
                if user_input.strip() == "pick2":
                    move_to_pick2()

                elif user_input.strip() == "place2":
                    move_to_place2()

                elif user_input.strip() == "pick":
                    move_to_pick()

                elif user_input.strip() == "place":
                    move_to_place()

                elif user_input.startswith("p"):
                    _, x, y, z, qx, qy, qz, qw = user_input.split()
                    x, y, z, qx, qy, qz, qw = map(float, [x, y, z, qx, qy, qz, qw])
                    move_cartesian_path(x, y, z, qx, qy, qz, qw)

                elif user_input.startswith("j"):
                    _, j1, j2, j3, j4, j5, j6, j7 = user_input.split()
                    joint_positions = list(map(float, [j1, j2, j3, j4, j5, j6, j7]))
                    move_to_joint_positions(joint_positions)

                elif user_input.strip() == "aruco":
                    
                    if aruco_pose is not None:
                        rospy.loginfo("Moving to the latest ArUco marker pose...")
                        move_cartesian_path(
                            aruco_pose.position.x,
                            aruco_pose.position.y,
                            aruco_pose.position.z,
                            aruco_pose.orientation.x,
                            aruco_pose.orientation.y,
                            aruco_pose.orientation.z,
                            aruco_pose.orientation.w,
                        )
                    else:
                        rospy.logwarn("No ArUco marker pose received yet.")
                elif user_input.strip() == "og":
                    call_empty_service("/wam/bhand/open_grasp")

                elif user_input.strip() == "os":
                    call_empty_service("/wam/bhand/open_spread")

                elif user_input.strip() == "cg":
                    call_empty_service("/wam/bhand/close_grasp")

                elif user_input.strip() == "cs":
                    call_empty_service("/wam/bhand/close_spread")
                
                elif user_input.strip() == "h":
                    go_home()

                elif user_input.strip() == "find":
                    find_and_move()
                
                elif user_input.strip() == "c":
                    # Get the current pose
                    current_pose = move_group.get_current_pose(end_effector_link).pose
                    rospy.loginfo("Current Cartesian Pose:")
                    rospy.loginfo(f"Position: x={current_pose.position.x}, y={current_pose.position.y}, z={current_pose.position.z}")
                    rospy.loginfo(f"Orientation: x={current_pose.orientation.x}, y={current_pose.orientation.y}, z={current_pose.orientation.z}, w={current_pose.orientation.w}")
                
                elif user_input.strip() == "go":
                    go_to_pickup()
                
                elif user_input.strip() == "ret":
                    go_to_return()
                    
                elif user_input.strip() == "goF":
                    send_velocity_command()

                elif user_input.strip() == "goB":
                    send_velocity_command2()
                
                elif user_input.strip() == "store":
                    store_bottle()
                    
                else:
                    rospy.logwarn("Invalid input. Try again.")
            except ValueError:
                rospy.logwarn("Invalid input format. Ensure numbers are provided in the correct format.")
            except rospy.ROSInterruptException:
                rospy.loginfo("Interrupted. Exiting...")
                break
    except KeyboardInterrupt:
        rospy.loginfo("Script terminated by user.")
