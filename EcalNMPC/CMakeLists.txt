cmake_minimum_required(VERSION 3.22)
project(nmpc VERSION 0.0.1 LANGUAGES CXX)

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
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Build type is RelWithDebInfo")
    # add_compile_options(-O1)
    set(CMAKE_CXX_FLAG "-O1")
elseif (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    message(STATUS "Build type is MinSizeRel")
    # add_compile_options(-Os)
    set(CMAKE_CXX_FLAG "-Os")
else ()
    message(STATUS "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

## 3) Protobuf：生成 C++ 代码
find_package(Protobuf REQUIRED)
file(GLOB_RECURSE PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/protos/*.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
add_library(my_proto_msgs ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(my_proto_msgs PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(my_proto_msgs PUBLIC protobuf::libprotobuf)
######################################################

find_package(casadi QUIET CONFIG)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(eCAL REQUIRED)

file(GLOB_RECURSE CPP_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
add_library(${PROJECT_NAME} ${CPP_SOURCE_FILES})

target_compile_features(${PROJECT_NAME} PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME}
    casadi::casadi
    Eigen3::Eigen
    yaml-cpp
    eCAL::core
    my_proto_msgs
)

#enable_testing()
add_subdirectory(apps)
