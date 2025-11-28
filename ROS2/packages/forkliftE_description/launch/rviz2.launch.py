import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_name = "forkliftE_description"
    urdf_name = "ForkliftE.urdf"

    pkg_path = get_package_share_directory(package_name=package_name)
    urdf_model_path = os.path.join(pkg_path, f"urdf/{urdf_name}")
    rviz_config_path = os.path.join(pkg_path, "config/perception.rviz")

    # robot_state_publisher_node = Node(
    #     package="robot_state_publisher",
    #     executable="robot_state_publisher", emulate_tty=True,

    #     arguments=['--ros-args', '--log-level', 'warn']
    # )

    # joint_state_publisher_node = Node(
    #     package="joint_state_publisher",
    #     executable="joint_state_publisher", emulate_tty=True,
    #     arguments=[urdf_model_path, '--ros-args', '--log-level', 'warn']
    # )

    with open(urdf_model_path, "r") as f:
        robot_desc = f.read()

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        emulate_tty=True,
        parameters=[{"robot_description": robot_desc}],
        arguments=['--ros-args', '--log-level', 'warn'],
    )

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        emulate_tty=True,
        parameters=[{"source_list": ["/forklift/joint_states"], "robot_description": robot_desc}],
        arguments=['--ros-args', '--log-level', 'warn'],
    )

    # "screen", "log"
    rviz2_node = Node(
        package="rviz2", executable="rviz2", output="screen", emulate_tty=True,
        # parameters=[{'use_sim_time': True}],
        arguments=["-d", rviz_config_path, '--ros-args', '--log-level', 'warn']
    )

    ld = LaunchDescription()
    ld.add_action(robot_state_publisher_node)
    ld.add_action(joint_state_publisher_node)
    ld.add_action(rviz2_node)

    return ld
