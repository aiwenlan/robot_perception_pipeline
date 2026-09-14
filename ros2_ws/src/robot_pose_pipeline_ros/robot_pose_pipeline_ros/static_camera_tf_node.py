from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from .pose_utils import fill_transform_message


class StaticCameraTF(Node):
    def __init__(self):
        super().__init__("static_camera_tf")
        self.declare_parameter("pose_file", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        path = Path(self.get_parameter("pose_file").value).expanduser()
        transform = np.loadtxt(path) if path.is_file() else np.eye(4)
        message = TransformStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.get_parameter("base_frame").value
        message.child_frame_id = self.get_parameter("camera_frame").value
        fill_transform_message(message.transform, transform)
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(message)
        self.get_logger().info(f"published T_base_camera from {path if path.is_file() else 'identity fallback'}")


def main(args=None):
    rclpy.init(args=args)
    node = StaticCameraTF()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
