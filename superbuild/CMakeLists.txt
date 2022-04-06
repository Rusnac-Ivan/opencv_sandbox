#
# === CMake lists for the mlpack_sandbox ===
#


# === CMake options ===

# Set CMake version
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

#
# Configure CMake environment
#

# Declare project
project(opencv-superbuild LANGUAGES C CXX)


# Set standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include(ExternalProject)

if(NOT DEFINED LIBRARIES_PREFIX)
    set(LIBRARIES_PREFIX ${CMAKE_BINARY_DIR})
endif()

if(NOT DEFINED LIBRARIES_INSTALL_PREFIX)
    set(LIBRARIES_INSTALL_PREFIX ${LIBRARIES_PREFIX}/installed/${CMAKE_SYSTEM_NAME})
endif()

#[[ExternalProject_Add(glfw
    PREFIX                  ${LIBRARIES_PREFIX}/glfw
    GIT_REPOSITORY          https://github.com/glfw/glfw.git
    GIT_TAG                 3.3.4
    CMAKE_ARGS              -DCMAKE_INSTALL_PREFIX:PATH=${LIBRARIES_INSTALL_PREFIX}/glfw
    CMAKE_CACHE_ARGS        -DGLFW_BUILD_DOCS:BOOL=OFF
                            -DGLFW_BUILD_TESTS:BOOL=OFF
                            -DGLFW_BUILD_EXAMPLES:BOOL=OFF
                            
    UPDATE_DISCONNECTED     TRUE
    UPDATE_COMMAND          ""
    BUILD_ALWAYS            FALSE
)]]

#[[ExternalProject_Add(imgui
    PREFIX                  ${LIBRARIES_PREFIX}/imgui
    DEPENDS                 glfw
    GIT_REPOSITORY          https://github.com/ocornut/imgui.git
    GIT_TAG                 v1.87
    UPDATE_DISCONNECTED     TRUE
    UPDATE_COMMAND          ""
    CONFIGURE_COMMAND       ""
    BUILD_COMMAND           ""
    INSTALL_COMMAND         ""
)
ExternalProject_Get_property(imgui SOURCE_DIR)
set(IMGUI_SOURCE_DIR ${SOURCE_DIR})]]

ExternalProject_Add(opencv
    PREFIX                  ${LIBRARIES_PREFIX}/opencv
    GIT_REPOSITORY          https://github.com/opencv/opencv.git
    GIT_TAG                 4.5.5
    CMAKE_ARGS              -DCMAKE_INSTALL_PREFIX:PATH=${LIBRARIES_INSTALL_PREFIX}/opencv
    CMAKE_CACHE_ARGS        -DBUILD_SHARED_LIBS:BOOL=OFF
                            -DBUILD_TESTS:BOOL=OFF
                            -DBUILD_PERF_TESTS:BOOL=OFF
                            -DBUILD_PERF_TESTS:BOOL=OFF
                            -DBUILD_EXAMPLES:BOOL=OFF
                            -DBUILD_opencv_apps:BOOL=OFF
                            -DBUILD_opencv_python2:BOOL=OFF
                            -DBUILD_opencv_python3:BOOL=OFF
                            -DBUILD_JAVA:BOOL=OFF
                            -DBUILD_FAT_JAVA_LIB:BOOL=OFF
                            
                            
    UPDATE_DISCONNECTED     TRUE
    UPDATE_COMMAND          ""
    BUILD_ALWAYS            FALSE
)

ExternalProject_Add(opencv_sandbox
    PREFIX              ${LIBRARIES_PREFIX}/opencv_sandbox
    DEPENDS             opencv
    SOURCE_DIR          ${PROJECT_SOURCE_DIR}/../
    CMAKE_ARGS          
    CMAKE_CACHE_ARGS    -DCMAKE_PREFIX_PATH:STRING=${LIBRARIES_INSTALL_PREFIX}/opencv
                        
    BUILD_ALWAYS        TRUE
    INSTALL_COMMAND     ""
)
