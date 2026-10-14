cmake_minimum_required(VERSION 3.8)
project(prana_mujoco_drive)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/prana_mujoco_teleop.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
