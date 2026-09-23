from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='object_detection',
            executable='video_publisher_node',
            # arguments=['--ros-args', '--log-level', 'debug']

        ),
        Node(
            package='object_detection',
            executable='video_object_detector',
            arguments=['--ros-args', '--log-level', 'debug']
        ),
        Node(
            package='object_detection',
            executable='visualizer_node',
            # arguments=['--ros-args', '--log-level', 'debug']
        ),
        Node(
            package='object_detection',
            executable='depth_estimation_node',
            arguments=['--ros-args', '--log-level', 'debug']
        )
    ])