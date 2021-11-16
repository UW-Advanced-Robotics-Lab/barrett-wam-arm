# BarretWamArm
Meta Package for tools to use our Barrett Wam Arm in simulation and with hardware.

# Launch in Simulation

Simulation tools were developed to improve and/or debug software packages used in demos and/or grasping experiments. 

1. First, launch Gazebo.

    ```roslaunch barrett_wam_arm_control barrett_wam_arm_control.launch```
    
    ```roslaunch barrett_wam_arm_sim empty_world.launch```
    
2. Launch TF Publisher.

    ```roslaunch tf_publisher tf_publisher_sim.launch```
    
3. Afterwards, RGB-D images were played back from the [ARL AffPose Dataset](https://github.com/UW-Advanced-Robotics-Lab/arl-affpose-dataset-utils). A 6-DoF pose was used to command the robot using trac ik in simulation.

    ```roslaunch trac_ik barrett_trac_ik_arl_affpose_simple.launch```

# Launch for ArUco Demo
1. First, ensure that the Barrett WAM arm is calibrated. 

    ```bt-zero-cal```

2. Launch the WAM node.

    ```roslaunch wam_node wam_node.launch```

3. Launch ZED camera.

    ```roslaunch zed_ros_wrapper zed.launch```

4. Launch TF Publisher.

    ```roslaunch tf_publisher tf_publisher.launch```
    
5. Launch ArUco Node. 

    ```roslaunch aruco_ros single.launch```
    
6. Launch Trac IK. 

    ```roslaunch trac_ik demo.launch```
