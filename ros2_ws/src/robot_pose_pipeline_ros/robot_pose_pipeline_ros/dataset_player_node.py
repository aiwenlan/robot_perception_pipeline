from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

from .pose_utils import fill_pose_message


def _imread(path: Path, flags: int):
    """OpenCV-safe read for non-ASCII paths."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def depth_to_xyz(
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    depth_scale_to_m: float,
    mask: np.ndarray | None,
) -> np.ndarray:
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        depth_m = depth.astype(np.float64)
        if np.nanmax(depth_m) > 20.0:
            depth_m = depth_m * depth_scale_to_m
    else:
        depth_m = depth.astype(np.float64) * depth_scale_to_m
    valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= 5.0)
    if mask is not None:
        valid &= mask > 0
    v_idx, u_idx = np.where(valid)
    if len(u_idx) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    z = depth_m[v_idx, u_idx]
    x = (u_idx.astype(np.float64) - cx) * z / fx
    y = (v_idx.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def xyz_to_pointcloud2(points: np.ndarray, header: Header) -> PointCloud2:
    message = PointCloud2()
    message.header = header
    message.height = 1
    message.width = int(len(points))
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = message.point_step * message.width
    message.is_dense = True
    message.data = points.astype(np.float32).tobytes()
    return message


class DatasetPlayer(Node):
    def __init__(self):
        super().__init__("dataset_player")
        self.declare_parameter("manifest", "")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("mask_mode", "gt_mask")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("max_frames", 0)
        manifest = Path(self.get_parameter("manifest").value).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        self.base = manifest.parent
        self.records = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        max_frames = int(self.get_parameter("max_frames").value)
        if max_frames > 0:
            self.records = self.records[:max_frames]
        self.index = 0
        loop_param = self.get_parameter("loop").value
        if isinstance(loop_param, str):
            self.loop = loop_param.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.loop = bool(loop_param)
        self.bridge = CvBridge()
        self.camera_frame = self.get_parameter("camera_frame").value
        self.mask_mode = self.get_parameter("mask_mode").value
        self.rgb_pub = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)
        self.mask_pub = self.create_publisher(Image, "/perception/mask", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/perception/target_cloud", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/perception/pose_camera", 10)
        self.timer = self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.publish_next)

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.base / path).resolve()

    def publish_next(self):
        if self.index >= len(self.records):
            if self.loop and self.records:
                self.index = 0
                self.get_logger().info("dataset replay loop restart")
            else:
                self.get_logger().info("dataset replay finished")
                self.timer.cancel()
                return
        record = self.records[self.index]
        stamp = self.get_clock().now().to_msg()
        import cv2

        rgb = _imread(self.resolve(record["rgb"]), cv2.IMREAD_COLOR)
        depth = _imread(self.resolve(record["depth"]), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise FileNotFoundError(f"cannot read RGB/depth for frame {record['frame_id']}")
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        depth_encoding = "16UC1" if depth.dtype == np.uint16 else "32FC1"
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding=depth_encoding)
        for message in (rgb_msg, depth_msg):
            message.header.stamp = stamp
            message.header.frame_id = self.camera_frame
        info = CameraInfo()
        info.header = rgb_msg.header
        info.width, info.height = rgb.shape[1], rgb.shape[0]
        info.k = np.asarray(record["camera_matrix"], dtype=float).reshape(-1).tolist()
        info.p = [info.k[0], 0.0, info.k[2], 0.0, 0.0, info.k[4], info.k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info)

        mask = None
        mask_key = "gt_mask" if self.mask_mode == "gt_mask" else "predicted_mask"
        if record.get(mask_key):
            mask = _imread(self.resolve(record[mask_key]), cv2.IMREAD_GRAYSCALE)
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = rgb_msg.header
            self.mask_pub.publish(mask_msg)

        camera_matrix = np.asarray(record["camera_matrix"], dtype=float)
        depth_scale = float(record.get("depth_scale_to_m", 0.001))
        points = depth_to_xyz(depth, camera_matrix, depth_scale, mask)
        self.cloud_pub.publish(xyz_to_pointcloud2(points, rgb_msg.header))

        if record.get("predicted_pose"):
            transform = np.loadtxt(self.resolve(record["predicted_pose"]))
            pose = PoseStamped()
            pose.header = rgb_msg.header
            fill_pose_message(pose.pose, transform)
            self.pose_pub.publish(pose)
        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = DatasetPlayer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
