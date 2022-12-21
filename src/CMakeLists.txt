cmake_minimum_required(VERSION 3.12)
enable_testing()

# Add include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Library sources
file(GLOB sources "*.cpp" "*.h")
add_library(${PROJECT_NAME} SHARED ${sources})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Link dependencies
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${GTSAM_LIBS}
                      ${gtdynamics_LIBS})

# Add includes for source directories 'BEFORE' any other include paths so that
# the compiler uses dyne headers in our source directory instead of any
# previously installed dyne headers.
target_include_directories(
  ${PROJECT_NAME} BEFORE
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> # main dyne includes:
         $<INSTALL_INTERFACE:include/>
         $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> # config.h
)

# Install library
install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

# Process subdirs
add_subdirectory(tests)

# if(DE_BUILD_SCRIPTS) add_subdirectory(scripts) endif()
