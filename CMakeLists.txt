cmake_minimum_required(VERSION 3.12)
enable_testing()
project(
  dyne
  DESCRIPTION "Proprioceptive Legged Robot State Estimator"
  VERSION 1.0.0
  LANGUAGES CXX C)

add_compile_options(-faligned-new)

# Enforce c++11 standards
add_compile_options(-std=c++11) # CMake 3.1 and earlier
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_AUTHOR "Varun Agrawal")
set(PROJECT_AUTHOR_EMAIL "varunagrawal@gatech.edu")

### Find dependent libraries
# Find GTSAM components
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBS gtsam)

find_package(GTSAMCMakeTools)
include(GtsamBuildTypes)
include(GtsamTesting)
include(GtsamPrinting)

find_package(gtdynamics REQUIRED)
set(gtdynamics_LIBS gtdynamics)

message(STATUS "GTSAM Version: ${GTSAM_VERSION}")
message(STATUS "GTDynamics Version: ${gtdynamics_VERSION}")

# For unit tests and scripts.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")

print_build_options_for_target(${PROJECT_NAME})

# Boost is included with GTSAM so nothing to do here

### Add source code to build
add_subdirectory(src)

option(DYNE_BUILD_PYTHON_WRAPPER "Build Python wrapper" ON)
if(DYNE_BUILD_PYTHON_WRAPPER)
  add_subdirectory(python)
endif()
