cmake_minimum_required(VERSION 3.28)
project(perception LANGUAGES C CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# --------------------------------------------------------------------------------
# Find dependencies: TensorRT
# --------------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(TENSORRT_ROOT /usr)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS ${TENSORRT_ROOT}
    PATH_SUFFIXES include
)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64
)
if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "Could not find TensorRT. Please set TENSORRT_ROOT.")
endif ()
add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION ${TENSORRT_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
)

# --------------------------------------------------------------------------------
# Find ROS2 dependencies
# --------------------------------------------------------------------------------
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies() # Automaticly find all the packages mentioned in 'package.xml'

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
ament_auto_add_executable(perception ${SOURCE_FILES})
target_include_directories(perception PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
    ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(perception
    CUDA::cudart
    TensorRT::nvinfer
    ${OpenCV_LIBS}
)
target_compile_features(perception PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

# --------------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------------
ament_auto_package(INSTALL_TO_SHARE)