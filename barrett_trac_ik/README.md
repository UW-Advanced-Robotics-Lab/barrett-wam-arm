# Brief Instruction on How to Run ARUCO Demo
### Launch modules required
roslaunch wam_node wam_node.launch
roslaunch zed_wrapper zed.launch
roslaunch barrett_tf_publisher barrett_tf_publisher.launch
roslaunch aruco_ros single.launch
roslaunch barrett_trac_ik_barrett_trac_ik_demo.launch

### Run ARUCO Demo super node:
rosrun barrett_trac_ik aruco_demo.py

### Stubbing Commands:
rostopic pub /task_completion_flag_summit std_msgs/Int8 "data: 2"

