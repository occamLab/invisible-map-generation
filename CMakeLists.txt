cmake_minimum_required(VERSION 2.8.3)
project(invisible_map_generation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O2 -Wall -Wextra -Wpedantic)
endif()

find_package(catkin REQUIRED COMPONENTS)

catkin_package()

include_directories(
# #   include
  ${catkin_INCLUDE_DIRS}
# #   ${Eigen_INCLUDE_DIRS}
)

# # add_executable(calibrate src/calibrate.cc)
# # target_link_libraries(calibrate ${catkin_LIBRARIES})

# # add_dependencies(calibrate ${catkin_EXPORTED_TARGETS})
