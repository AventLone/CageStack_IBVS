cmake_minimum_required(VERSION 4.3)
project(perception LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()

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
find_package(PCL REQUIRED COMPONENTS common registration filters)
find_package(CUDAToolkit REQUIRED)
find_package(Sophus REQUIRED)

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cuCollections/include/cuco/static_map.cuh)
    message(FATAL_ERROR "cuCollections headers not found at ${CMAKE_CURRENT_SOURCE_DIR}/cuCollections/include")
endif ()
set(BUILD_TESTS OFF CACHE BOOL "Disable cuCollections tests" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable cuCollections benchmarks" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable cuCollections examples" FORCE)
set(INSTALL_CUCO OFF CACHE BOOL "Disable cuCollections install rules" FORCE)
add_subdirectory(cuCollections EXCLUDE_FROM_ALL)

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
list(REMOVE_ITEM SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/source/LIO/SophusPose.cpp)

add_library(sparsity_aware_gicp STATIC
    source/LIO/SparsityAwareGICP.cu
    source/LIO/SophusPose.cpp
)
target_include_directories(sparsity_aware_gicp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(sparsity_aware_gicp PRIVATE ${PCL_LIBRARIES} CUDA::cudart Sophus::Sophus cuco::cuco)
target_compile_features(sparsity_aware_gicp PUBLIC cxx_std_23 cuda_std_17)
target_compile_options(sparsity_aware_gicp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

ament_auto_add_executable(perception ${SOURCE_FILES})
target_include_directories(perception PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(perception ${PCL_LIBRARIES} sparsity_aware_gicp)
target_compile_features(perception PUBLIC c_std_99 cxx_std_23)  # Require C99 and C++23

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