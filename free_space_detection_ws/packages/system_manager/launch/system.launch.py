from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch.actions import TimerAction
import os


def generate_launch_description():
    this_package = get_package_share_directory("system_manager")
    system_params_file = os.path.join(this_package, 'config/param', 'system.yaml')

    perception_node = Node(package="perception", executable="perception", namespace="perception", 
                           emulate_tty=True, parameters=[system_params_file])
    control_node = Node(package="control", executable="control", emulate_tty=True, parameters=[system_params_file])

    robot_description_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        PathJoinSubstitution([this_package, 'launch','rviz2.launch.py'])))

    ld = LaunchDescription()
    ld.add_action(robot_description_launch)
    ld.add_action(TimerAction(period=1.6, actions=[perception_node, control_node]))
    return ld
