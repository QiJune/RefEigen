cmake_minimum_required(VERSION 3.0)
project(RefEigen)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PROJ_ROOT ${CMAKE_CURRENT_SOURCE_DIR})


find_package(Eigen3 REQUIRED)
find_package(CUDA QUIET)

include_directories(${EIGEN3_INCLUDE_DIR})

option(WITH_GPU         "Compile with NVIDIA GPU"          ${CUDA_FOUND})

if(NOT WITH_GPU)
  add_definitions(-DONLY_CPU)
endif()

set(SOURCE_FILES main.cc.cu)
if(WITH_GPU)
  message("WITH_GPU")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-std=c++11;-O2;-DVERBOSE")
  cuda_add_executable(RefEigen ${SOURCE_FILES})
else()
  add_executable(RefEigen ${SOURCE_FILES})
endif()
