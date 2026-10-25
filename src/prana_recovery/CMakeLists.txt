cmake_minimum_required(VERSION 3.8)
project(prana_recovery)

find_package(ament_cmake REQUIRED)

install(DIRECTORY scripts/prana_recovery
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/train_prana_recovery.py
  scripts/evaluate_prana_recovery.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
