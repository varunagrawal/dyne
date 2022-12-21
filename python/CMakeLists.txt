# Wrapping to Python

find_package(gtwrap)

set(WRAP_PYTHON_VERSION
    "3.8"
    CACHE STRING "The Python version to use for wrapping")

gtwrap_get_python_version(${WRAP_PYTHON_VERSION})
message(STATUS "Wrapper Python Version: ${WRAP_PYTHON_VERSION}")

set(PYTHON_REQUIREMENTS_PATH "requirements.txt")
# set(PYTHON_TESTS "${PROJECT_NAME}.tests")

set(DYNE_PYTHON_PATH ${PROJECT_SOURCE_DIR}/python/)
set(DYNE_MODULE_PATH ${PROJECT_SOURCE_DIR}/dyne/)

set(MODULE_NAME "state_estimator")

configure_file(${DYNE_PYTHON_PATH}/${PROJECT_NAME}.tpl
               ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.tpl)

# Copy the interface header file to the correct location
configure_file(${DYNE_PYTHON_PATH}/${PROJECT_NAME}.i
               ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.i COPYONLY)

# # Copy the dyne package directory
# file(COPY ${DYNE_PYTHON_PATH}/${PROJECT_NAME} DESTINATION ${DE_MODULE_PATH})

pybind_wrap(
  ${PROJECT_NAME}_py # target
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.i # interface header file
  "${PROJECT_NAME}.cpp" # the generated cpp
  "${MODULE_NAME}" # module_name
  "dyne" # top namespace in the cpp file
  "${ignore}" # ignore classes
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.tpl
  ${PROJECT_NAME} # libs
  "${PROJECT_NAME}" # dependencies
  ON # use boost
)

# Set the install location for the generated .so file
set_target_properties(
  ${PROJECT_NAME}_py
  PROPERTIES OUTPUT_NAME "${MODULE_NAME}"
             LIBRARY_OUTPUT_DIRECTORY "${DYNE_MODULE_PATH}"
             DEBUG_POSTFIX "" # Otherwise you will have a wrong name
             RELWITHDEBINFO_POSTFIX "" # Otherwise you will have a wrong name
)
