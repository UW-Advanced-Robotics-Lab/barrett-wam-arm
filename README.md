# BarretWamArm
Meta Package for tools to use our Barrett Wam Arm in simulation and with hardware.

# ROS Architecture
Here is an overview of our Architecture.
![ros_overview](https://user-images.githubusercontent.com/56097508/141863644-5be5be99-79fe-4f77-a17c-9e671ab23c45.png)

# Launch for ArUco Demo
1. First, ensure that the Barrett WAM arm is calibrated. 

    ```bt-zero-cal```

2. Launch the WAM node.

    ```roslaunch wam_node wam_node```

3. Launch ZED camera.

    ```roslaunch zed_ros_wrapper zed.launch```

4. Launch TF Publisher.

    ```roslaunch tf_publisher tf_publisher.launch```
    
5. Launch ArUco Node. 

    ```roslaunch aruco_ros single.launch```
    
6. Launch Trac IK. 

    ```roslaunch trac_ik demo.launch```
