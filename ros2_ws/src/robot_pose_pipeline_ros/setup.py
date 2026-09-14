from glob import glob
from setuptools import find_packages, setup

package_name = "robot_pose_pipeline_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="student",
    maintainer_email="student@example.com",
    description="Offline RGB-D, 6D pose and TF2 replay",
    license="MIT",
    entry_points={
        "console_scripts": [
            "dataset_player = robot_pose_pipeline_ros.dataset_player_node:main",
            "static_camera_tf = robot_pose_pipeline_ros.static_camera_tf_node:main",
            "pose_transform = robot_pose_pipeline_ros.pose_transform_node:main",
        ]
    },
)
