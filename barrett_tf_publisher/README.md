# Object Pose for Grasping

![ros_overview](../samples/coordinate_transforms.png)

There are four main coordinate frames are used to grasp an object.

1. Base link of the manipulator

2. Camera Frame

    This is a dynamic transform as we mounted our camera on our arm. **This package publishes the Camera Frame.**
     
3. Object Frame
    
    The 6-DoF pose was determined either using marker-based methods, such as [aruco_ros](https://github.com/pal-robotics/aruco_ros), or using deep learning, such as [DOPE](https://github.com/NVlabs/Deep_Object_Pose) or [DenseFusion](https://github.com/j96w/DenseFusion).
    
4. End Effector Frame

    We used a 8-DoF Barrett Hand. Which has +/- 17.5cm from tip to the center of the palm. Note that two-finger grippers require the object pose to be accurate within +/- 2cm.
    
Ultimately, we need to align the end effector frame to the pose of the object w.r.t. base link of the manipulator.