from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='object_detection',
            executable='video_publisher_node'
        ),
        Node(
            package='object_detection',
            executable='video_object_detector'
        ),
        Node(
            package='object_detection',
            executable='visualizer_node'
        )
    ])