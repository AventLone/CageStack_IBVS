# 查找目录中的所有 *.cpp 文件
file(GLOB TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 遍历每个 *.cpp 文件并创建对应的可执行文件
foreach (TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_TARGET ${TEST_FILE} NAME_WE) # 获取文件名（不包括路径和后缀）
    add_executable(${TEST_TARGET} ${TEST_FILE}) # 将 .cpp 文件作为可执行文件添加
    # target_include_directories(${TEST_TARGET} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(${TEST_TARGET} ${PROJECT_NAME})
    add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})
endforeach ()