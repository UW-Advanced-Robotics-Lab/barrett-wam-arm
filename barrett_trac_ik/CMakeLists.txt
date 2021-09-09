cmake_minimum_required(VERSION 2.8.3)
project(barrett_trac_ik)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

add_service_files(DIRECTORY srv FILES JointMove.srv PoseMove.srv OpenGrasp.srv CloseGrasp.srv)
generate_messages(DEPENDENCIES std_msgs geometry_msgs)

catkin_package()

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

catkin_install_python(PROGRAMS
  scripts/barrett_trac_ik_arl_affpose_autonomous.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/barrett_trac_ik
)

