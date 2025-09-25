cmake_minimum_required(VERSION 3.22)
project(CommonLib VERSION 0.0.1 LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Build type is Debug")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build type is Release")
    # add_compile_options(-O2)
    # set(CMAKE_CXX_FLAG "-O2")
    set(CMAKE_CXX_FLAG "-O3")
    set(CMAKE_CUDA_FLAGS "-O2")  # 确保只有一个优化等级
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Build type is RelWithDebInfo")
    # add_compile_options(-O1)
    set(CMAKE_CXX_FLAG "-O1")
    set(CMAKE_CUDA_FLAGS "-O1")  # 确保只有一个优化等级
elseif (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    message(STATUS "Build type is MinSizeRel")
    # add_compile_options(-Os)
    set(CMAKE_CXX_FLAG "-Os")
    set(CMAKE_CUDA_FLAGS "-Os")  # 确保只有一个优化等级
else ()
    message(STATUS "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(casadi QUIET CONFIG)
find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
)

file(GLOB_RECURSE CPP_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
add_library(${PROJECT_NAME} ${CPP_SOURCE_FILES})

target_compile_features(${PROJECT_NAME} PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
    ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
    casadi::casadi
)

#enable_testing()
add_subdirectory(tests)
