# Deep End-to-end Motion Planning in ROS Stage Simulator

### About


This work aims at implementing end-to-end motion planning in ROS stage simulator which is a reproduction of the paper "Pfeiffer M , Schaeuble M , Nieto J , et al. From Perception to Decision: A Data-driven Approach to End-to-end Motion Planning for Autonomous Ground Robots[J]. 2016."

The deep neural network is a CNN with 2 ResNet block using the 1080-dimensional LiDAR measurement.The trainning is done in a supervised fashion using ROS navigation stack demonstrations.


### Dependencies

This code is based on [navigation_stage package](http://wiki.ros.org/navigation_stage) in ROS and has been tested on ROS kinetic. 

This demonstrative run is built on the pre-trained neural network model using Tensorflow in a supervised fashion using navigation stack in ROS, the code of which will be posted afterwards if anybody asks.

### How to Run

Please use `copy.files.bash` to copy stage simulator configuration files.

After using the `move_base_amcl_cnn.launch` to launch stage, the CNN node can be established using rospy to run the `scripts/cnn_node.py`

## Acknowledgement

This code set is our reproduction of the following ICRA2017 paper

```
Pfeiffer M , Schaeuble M , Nieto J , et al. From Perception to Decision: A Data-driven Approach to End-to-end Motion Planning for Autonomous Ground Robots[J]. 2016.
```




