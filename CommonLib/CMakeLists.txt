cmake_minimum_required(VERSION 3.22)
project(Perception3D VERSION 0.0.1 LANGUAGES C CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     add_compile_options(-Wall -Wextra -Wpedantic)
# endif()
add_definitions(-w)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Build type is Debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build type is Release")
    # add_compile_options(-O2)
    # set(CMAKE_CXX_FLAG "-O2")
    set(CMAKE_CXX_FLAG "-O3")
    set(CMAKE_CUDA_FLAGS "-O2")  # 确保只有一个优化等级
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Build type is RelWithDebInfo")
    # add_compile_options(-O1)
    set(CMAKE_CXX_FLAG "-O1")
    set(CMAKE_CUDA_FLAGS "-O1")  # 确保只有一个优化等级
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    message(STATUS "Build type is MinSizeRel")
    # add_compile_options(-Os)
    set(CMAKE_CXX_FLAG "-Os")
    set(CMAKE_CUDA_FLAGS "-Os")  # 确保只有一个优化等级
else()
    message(STATUS "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(PCL REQUIRED)
# find_package(eCAL REQUIRED)
# find_package(Boost REQUIRED COMPONENTS log)


include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${PARENT_DIR}/Common/include
)

file(GLOB_RECURSE CPP_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
add_library(${PROJECT_NAME} STATIC ${CPP_SOURCE_FILES})

file(GLOB_RECURSE CUDA_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cu)
add_library(cuda_lib STATIC ${CUDA_SOURCE_FILES})

set(DEPENDENT_LIBS
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    # eCAL::core
    ${PROJECT_NAME}
    cuda_lib
)

add_subdirectory(apps)

enable_testing()
add_subdirectory(tests)
