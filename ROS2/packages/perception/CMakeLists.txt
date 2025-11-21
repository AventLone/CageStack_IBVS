cmake_minimum_required(VERSION 3.28)
project(perception LANGUAGES C CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

find_package(OpenCV REQUIRED)

# --------------------------------------------------------------------------------
# Find ROS2 dependencies
# --------------------------------------------------------------------------------
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies() # Automaticly find all the packages mentioned in 'package.xml'

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
ament_auto_add_executable(perception ${SOURCE_FILES})
target_include_directories(perception PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_compile_features(perception PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(perception
    ${OpenCV_LIBS}
)

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

# --------------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------------
ament_auto_package(INSTALL_TO_SHARE launch config)