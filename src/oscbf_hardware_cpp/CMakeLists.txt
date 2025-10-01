cmake_minimum_required(VERSION 3.8)
project(oscbf_libfranka_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Franka REQUIRED)
# find_package(pinocchio REQUIRED)

# Define executable
add_executable(franka_impedance_controller src/franka_impedance_controller.cpp)

# Link dependencies
target_include_directories(franka_impedance_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(franka_impedance_controller
  franka
)

ament_target_dependencies(franka_impedance_controller
  rclcpp
  sensor_msgs
  std_msgs
)

install(TARGETS franka_impedance_controller
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
