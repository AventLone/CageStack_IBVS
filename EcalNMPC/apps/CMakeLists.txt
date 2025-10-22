# 查找目录中的所有 *.cpp 文件
file(GLOB TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 遍历每个 *.cpp 文件并创建对应的可执行文件
foreach (TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_TARGET ${TEST_FILE} NAME_WE) # 获取文件名（不包括路径和后缀）
    add_executable(${TEST_TARGET} ${TEST_FILE}) # 将 .cpp 文件作为可执行文件添加
    target_compile_definitions(${TEST_TARGET} PRIVATE PARAM_PATH="${PARENT_DIR}/configs/nmpc_st.yaml")
    target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)  # Require C99 and C++17
    target_link_libraries(${TEST_TARGET} ${PROJECT_NAME})
endforeach ()