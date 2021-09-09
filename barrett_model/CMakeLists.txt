cmake_minimum_required(VERSION 2.8.3)
project(barrett_model)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  barrett_wam_arm_hw
)

catkin_package(
  CATKIN_DEPENDS
    barrett_wam_arm_hw
)
