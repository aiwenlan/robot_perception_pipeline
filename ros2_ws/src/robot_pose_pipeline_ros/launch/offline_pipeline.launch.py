from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    manifest = LaunchConfiguration("manifest")
    base_camera_pose = LaunchConfiguration("base_camera_pose")
    mask_mode = LaunchConfiguration("mask_mode")
    loop = LaunchConfiguration("loop")
    max_frames = LaunchConfiguration("max_frames")
    rate_hz = LaunchConfiguration("rate_hz")
    return LaunchDescription([
        DeclareLaunchArgument("manifest", description="Absolute JSONL manifest path"),
        DeclareLaunchArgument("base_camera_pose", default_value="", description="4x4 T_base_camera text file; empty uses identity"),
        DeclareLaunchArgument("mask_mode", default_value="gt_mask", description="gt_mask or predicted_mask"),
        DeclareLaunchArgument("loop", default_value="false", description="Loop dataset replay"),
        DeclareLaunchArgument("max_frames", default_value="0", description="Limit frames; 0 means all"),
        DeclareLaunchArgument("rate_hz", default_value="5.0", description="Replay rate"),
        Node(package="robot_pose_pipeline_ros", executable="static_camera_tf", parameters=[{"pose_file": base_camera_pose}]),
        Node(
            package="robot_pose_pipeline_ros",
            executable="dataset_player",
            parameters=[
                {
                    "manifest": manifest,
                    "mask_mode": mask_mode,
                    "loop": loop,
                    "max_frames": max_frames,
                    "rate_hz": rate_hz,
                }
            ],
        ),
        Node(package="robot_pose_pipeline_ros", executable="pose_transform"),
    ])
