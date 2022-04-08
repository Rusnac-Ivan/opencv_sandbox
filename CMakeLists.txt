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
# Set src and include root path
set(LIBS_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/")

file(
    GLOB_RECURSE SOURCE_FILES
    LIST_DIRECTORIES false
    "${SRC_ROOT_PATH}/*.c"
    "${SRC_ROOT_PATH}/*.cpp"
    "${SRC_ROOT_PATH}/*.h"
    "${SRC_ROOT_PATH}/*.hpp"
)

file(
    GLOB_RECURSE LIBS_FILES
    LIST_DIRECTORIES false
    "${LIBS_ROOT_PATH}/*.c"
    "${LIBS_ROOT_PATH}/*.cpp"
    "${LIBS_ROOT_PATH}/*.h"
    "${LIBS_ROOT_PATH}/*.hpp"
)

foreach(_source IN ITEMS ${SOURCE_FILES})
    get_filename_component(_source_path "${_source}" PATH)
    file(RELATIVE_PATH _source_path_rel "${SRC_ROOT_PATH}" "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
    source_group("source\\${_group_path}" FILES "${_source}")
endforeach()

foreach(_source IN ITEMS ${LIBS_FILES})
    get_filename_component(_source_path "${_source}" PATH)
    file(RELATIVE_PATH _source_path_rel "${LIBS_ROOT_PATH}" "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
    source_group("source\\${_group_path}" FILES "${_source}")
endforeach()

set(GLAD_INCLUDE_DIR "${LIBS_ROOT_PATH}/glad-0.1.34/include")

#file(GLOB GLAD_SRC "${LIBS_ROOT_PATH}/glad-0.1.34/src/*.c")

add_executable(${PROJECT_NAME} ${INCLUDE_LIST} ${SOURCE_FILES} ${LIBS_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ ${GLAD_INCLUDE_DIR})

find_package( OpenCV REQUIRED core face features2d videoio ccalib calib3d)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )

find_package(GLFW3 REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${GLFW3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${GLFW3_LIBRARY})
message(STATUS "GLFW3_INCLUDE_DIR: ${GLFW3_INCLUDE_DIR}")
message(STATUS "GLFW3_LIBRARY: ${GLFW3_LIBRARY}")

if(IMGUI_PATH)
    #message(STATUS "IMGUI_PATH: ${IMGUI_PATH}")
    set(IMGUI_SRC 
            ${IMGUI_PATH}/imconfig.h
            ${IMGUI_PATH}/imgui.cpp
            ${IMGUI_PATH}/imgui.h
            ${IMGUI_PATH}/imgui_demo.cpp
            ${IMGUI_PATH}/imgui_draw.cpp
            ${IMGUI_PATH}/imgui_internal.h
            ${IMGUI_PATH}/imgui_tables.cpp
            ${IMGUI_PATH}/imgui_widgets.cpp
            ${IMGUI_PATH}/imstb_rectpack.h
            ${IMGUI_PATH}/imstb_textedit.h
            ${IMGUI_PATH}/imstb_truetype.h
            ${IMGUI_PATH}/backends/imgui_impl_opengl3.cpp
            ${IMGUI_PATH}/backends/imgui_impl_opengl3.h
            ${IMGUI_PATH}/backends/imgui_impl_glfw.cpp
            ${IMGUI_PATH}/backends/imgui_impl_glfw.h
    )
    
    message(STATUS "IMGUI_SRC: ${IMGUI_SRC}")
    target_sources(${PROJECT_NAME} PRIVATE ${IMGUI_SRC})
    target_include_directories(${PROJECT_NAME} PRIVATE ${IMGUI_PATH})
else()
    message(FATAL_ERROR "IMGUI_PATH is not set!")
endif()

find_package(glm REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)
