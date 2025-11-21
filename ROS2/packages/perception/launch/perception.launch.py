import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
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
    ld.add_action(rviz2_node)

    return ld
