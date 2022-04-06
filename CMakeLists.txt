#
# === CMake lists for the mlpack_sandbox ===
#


# === CMake options ===

# Set CMake version
cmake_minimum_required(VERSION 3.15 FATAL_ERROR) 

project(opencv_sandbox LANGUAGES C CXX)

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set src and include root path
set(SRC_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/")

file(
    GLOB_RECURSE SOURCE_FILES
    LIST_DIRECTORIES false
    "${SRC_ROOT_PATH}/*.c"
    "${SRC_ROOT_PATH}/*.cpp"
    "${SRC_ROOT_PATH}/*.h"
    "${SRC_ROOT_PATH}/*.hpp"
)

foreach(_source IN ITEMS ${SOURCE_FILES})
    get_filename_component(_source_path "${_source}" PATH)
    file(RELATIVE_PATH _source_path_rel "${SRC_ROOT_PATH}" "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
    source_group("source\\${_group_path}" FILES "${_source}")
endforeach()

add_executable(${PROJECT_NAME} ${INCLUDE_LIST} ${SOURCE_FILES})

find_package( OpenCV REQUIRED )
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
