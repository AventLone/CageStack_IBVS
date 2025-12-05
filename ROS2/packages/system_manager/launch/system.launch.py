from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import math, os


def generate_launch_description():
    this_package = get_package_share_directory("system_manager")
    system_params_file = os.path.join(this_package, 'config/param', 'system.yaml')

    perception_node = Node(package="perception", executable="perception", namespace="perception", 
                           emulate_tty=True, parameters=[system_params_file])
    control_node = Node(package="control", executable="control", emulate_tty=True, parameters=[system_params_file])

    tf2_node = Node(package="tf2_ros",
                    executable="static_transform_publisher",
                    output="screen",
                    emulate_tty=True,
                    arguments=[
                        '--x', '0.0', '--y', '0.0', '--z', '0.0',
                        '--roll', '0.0', '--pitch', '0.0', '--yaw', f'{-math.pi * 0.5}',
                        '--frame-id', 'map', '--child-frame-id', 'Forklift_E',
                        '--ros-args', '--log-level', 'warn'
                    ])

    robot_description_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        PathJoinSubstitution([this_package, 'launch','rviz2.launch.py'])))

    ld = LaunchDescription()
    ld.add_action(perception_node)
    ld.add_action(control_node)
    ld.add_action(tf2_node)
    ld.add_action(robot_description_launch)

    return ld
