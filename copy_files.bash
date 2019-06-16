#!/bin/bash
source /opt/ros/kinetic/setup.bash

nav_des=`rospack find navigation_stage`
launch_dir=${nav_des}"/launch"
map_dir=${nav_des}"/stage_config"
cp move_base_amcl_cnn.launch $launch_dir
cp -r tur $map_dir

