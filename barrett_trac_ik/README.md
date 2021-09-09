# Barrett WAM Arm tf publisher

In this work we want to transform the object's pose from the camera frame, a measurement provided by [DenseFusion](https://github.com/j96w/DenseFusion), to the base link of our 7-DoF arm for robotic grasping. 

I use this TF Publisher with the following repos:

1. [Labelusion](https://github.com/akeaveny/LabelFusion) for generating Real Images
2. [NDDS](https://github.com/NVIDIA/Dataset_Synthesizer) for generating Synthetic Images
3. [PyTorch-Simple-AffNet](https://github.com/akeaveny/DenseFusion) for predicting 6-DoF Object Pose.
4. [DenseFusion](https://github.com/akeaveny/DenseFusion) for predicting 6-DoF Object Pose.
5. AffDenseFusionROSNode: coming soon.

Here is a demo of this repo in simulation.
![Alt text](samples/sim_demo.gif?raw=true "Title")

Here is a demo of us grasping an object in the lab!
![Alt text](samples/lab_demo.gif?raw=true "Title")
