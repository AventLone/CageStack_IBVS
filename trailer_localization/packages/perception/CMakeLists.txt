cmake_minimum_required(VERSION 3.22)
project(perception LANGUAGES C CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
endif ()

# --------------------------------------------------------------------------------
# Find ROS2 dependencies
# --------------------------------------------------------------------------------
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies() # Automaticly find all the packages mentioned in 'package.xml'

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)

find_package(Sophus REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost 1.81 REQUIRED)
find_package(Threads REQUIRED)

ament_auto_add_executable(perception ${SOURCE_FILES})
target_include_directories(perception PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${Boost_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(perception Sophus::Sophus TBB::tbb Threads::Threads)
target_compile_features(perception PUBLIC c_std_99 cxx_std_23)

if (BUILD_TESTING)
    add_executable(eskf_test test/eskf_test.cpp source/LIO/ESKF.cpp source/LIO/SparseVoxel.cpp)
    target_include_directories(eskf_test PRIVATE include ${Boost_INCLUDE_DIRS})
    target_link_libraries(eskf_test Sophus::Sophus TBB::tbb)
    ament_target_dependencies(eskf_test pcl_ros)
    target_compile_features(eskf_test PRIVATE cxx_std_23)
    add_test(NAME eskf_test COMMAND eskf_test)

    add_executable(lio_measurement_test test/lio_measurement_test.cpp
        source/LIO/SparseVoxel.cpp source/LIO/ESKF.cpp)
    target_include_directories(lio_measurement_test PRIVATE include ${Boost_INCLUDE_DIRS})
    target_link_libraries(lio_measurement_test Sophus::Sophus TBB::tbb)
    ament_target_dependencies(lio_measurement_test pcl_ros sensor_msgs)
    target_compile_features(lio_measurement_test PRIVATE cxx_std_23)
    add_test(NAME lio_measurement_test COMMAND lio_measurement_test)

    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

# --------------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------------
ament_auto_package(INSTALL_TO_SHARE config docs)
