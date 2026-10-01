from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='person_pub',
            executable='bodyposenet_onnx_node',
            output='screen',
        ),
        Node(
            package='person_pub',
            executable='depth_fusion',
            output='screen',
            parameters=[{
                'depth_info_topic': '/depth/camera_info',
                'depth_image_topic': '/depth',
                'compressed': False,
            }],
        ),
        Node(
            package='person_pub',
            executable='pose_filter_node',
            output='screen',
        ),
        Node(
            package='person_pub',
            executable='person_pub',
            output='screen',
        ),
    ])
