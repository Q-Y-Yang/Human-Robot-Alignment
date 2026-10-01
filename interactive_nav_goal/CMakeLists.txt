cmake_minimum_required(VERSION 3.8)
project(interactive_nav_goal)

find_package(ament_cmake REQUIRED)

# Keep the node's implementation at the workspace src/ level while installing it
# as a normal ROS 2 executable.
install(
  PROGRAMS
    scripts/robot_pose_align_to_human.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME robot_pose_align_to_human
)

# This helper is imported by robot_pose_align_to_human.py at runtime.
install(
  FILES
    scripts/fformations_compact.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
