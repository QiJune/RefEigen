cmake_minimum_required(VERSION 3.0)
project(RefEigen)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PROJ_ROOT ${CMAKE_CURRENT_SOURCE_DIR})


find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)
find_package(Eigen3 REQUIRED)

include(generic)
include_directories(${EIGEN3_INCLUDE_DIR})

set(SOURCE_FILES main.cpp)
add_executable(RefEigen ${SOURCE_FILES})