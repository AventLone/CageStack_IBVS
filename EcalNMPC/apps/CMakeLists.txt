# 查找目录中的所有 *.cpp 文件
file(GLOB APP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 遍历每个 *.cpp 文件并创建对应的可执行文件
foreach (APP_FILE ${APP_FILES})
    get_filename_component(APP_TARGET ${APP_FILE} NAME_WE) # 获取文件名（不包括路径和后缀）
    add_executable(${APP_TARGET} ${APP_FILE}) # 将 .cpp 文件作为可执行文件添加
    target_compile_definitions(${APP_TARGET} PRIVATE CONFIG_FILES_DIR="${CMAKE_SOURCE_DIR}/configs/")
    target_include_directories(${APP_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(${APP_TARGET} PUBLIC cxx_std_17)  # Require C99 and C++17
    target_link_libraries(${APP_TARGET} ${PROJECT_NAME})
endforeach ()