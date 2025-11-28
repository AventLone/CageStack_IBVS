import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # 定义可配置的参数
#     pointcloud_topic_arg = DeclareLaunchArgument(
#         'pointcloud_topic',
#         default_value='/cloud',
#         description='输入点云的话题名称'
#     )
#
#     frame_id_arg = DeclareLaunchArgument(
#         'frame_id',
#         default_value='map',
#         description='OctoMap的坐标系'
#     )
#
#     resolution_arg = DeclareLaunchArgument(
#         'resolution',
#         default_value='0.05',
#         description='OctoMap的分辨率（米）'
#     )
#
#     max_range_arg = DeclareLaunchArgument(
#         'max_range',
#         default_value='-1.0',
#         description='最大感知范围'
#     )
#
#     publish_free_space_arg = DeclareLaunchArgument(
#         'publish_free_space',
#         default_value='false',
#         description='是否发布自由空间'
#     )
    
    # 启动octomap_server节点
#     octomap_server_node = Node(
#         package='octomap_server',
#         executable='octomap_server_node',
#         name='octomap_server',
#         output='screen',
#         emulate_tty=True,
#         parameters=[{
#             'frame_id': LaunchConfiguration('frame_id'),
#             'resolution': LaunchConfiguration('resolution'),
#             'sensor_model.max_range': LaunchConfiguration('max_range'),
#             'publish_free_space': LaunchConfiguration('publish_free_space'),
#
#             # 点云预处理参数
#             'pointcloud_min_z': 0.01,
#             'pointcloud_max_z': 10.0,
#
#             # OctoMap参数
#             'occupancy_min_z': 0.1,
#             'occupancy_max_z': 3.0,
#             'filter_ground': True,
#             'ground_filter.distance': 0.04,
#             'ground_filter.angle': 0.15,
#             'ground_filter.plane_distance': 0.07,
#
#             # 发布配置
#             'publish_2d_map': True,  # 同时发布2D占据栅格
#         }],
#         remappings=[
#             ('cloud_in', LaunchConfiguration('pointcloud_topic')),
#             ('octomap_full', '/octomap_full'),
#             ('octomap_binary', '/octomap_binary'),
#             ('projected_map', '/map')
#         ]
#     )
    octomap_server_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'frame_id': "map",
            'resolution': 0.01,
            'filter_ground': True,
            'ground_filter.distance': 0.04,
            'ground_filter.angle': 0.15,
            'ground_filter.plane_distance': 0.07,
            'publish_2d_map': True,  # 同时发布2D占据栅格
        }],
        remappings=[
            ('cloud_in', "/cloud"),
            ('octomap_full', '/octomap_full'),
            ('octomap_binary', '/octomap_binary'),
            ('projected_map', '/map')
        ]
    )

    package_name = "perception"

    pkg_path = get_package_share_directory(package_name=package_name)
    rviz_config_path = os.path.join(pkg_path, "config/perception.rviz")

    perception_node = Node(package=package_name, executable="perception", emulate_tty=True)

    # "screen", "log"
    rviz2_node = Node(
        package="rviz2", executable="rviz2", output="screen", emulate_tty=True,
        parameters=[{'use_sim_time': True}],
        arguments=["-d", rviz_config_path, '--ros-args', '--log-level', 'warn']
    )

    ld = LaunchDescription()
    ld.add_action(perception_node)
    ld.add_action(octomap_server_node)
    ld.add_action(rviz2_node)

    return ld