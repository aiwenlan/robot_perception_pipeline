"""Optional ROS2 node: publish processed / object / CAD-aligned clouds.

Does not replace dataset_player. Subscribe to RGB-D + mask + pose_camera.

  ros2 run robot_pose_pipeline_ros cloud_processor --ros-args -p model_points:=/path/points.txt
"""

from __future__ import annotations

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header


def xyz_to_cloud(points: np.ndarray, header: Header) -> PointCloud2:
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(len(points))
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = np.asarray(points, dtype=np.float32).tobytes()
    return msg


class CloudProcessor(Node):
    def __init__(self):
        super().__init__("cloud_processor")
        self.bridge = CvBridge()
        self.declare_parameter("model_points", "")
        self.declare_parameter("voxel_size", 0.005)
        self.declare_parameter("depth_scale_to_m", 0.001)
        self.rgb = None
        self.depth = None
        self.mask = None
        self.pose_T = None
        self.K = None
        self._header = Header()
        self.model_xyz = None
        model_path = self.get_parameter("model_points").value
        if model_path:
            try:
                self.model_xyz = np.loadtxt(model_path)
                self.get_logger().info(f"loaded model points: {self.model_xyz.shape}")
            except Exception as exc:
                self.get_logger().warn(f"model load failed: {exc}")

        self.pub_processed = self.create_publisher(PointCloud2, "/perception/processed_cloud", 10)
        self.pub_object = self.create_publisher(PointCloud2, "/perception/object_cloud", 10)
        self.pub_cad = self.create_publisher(PointCloud2, "/perception/cad_aligned_cloud", 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_rgb, 10)
        self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, 10)
        self.create_subscription(Image, "/perception/mask", self.on_mask, 10)
        self.create_subscription(PoseStamped, "/perception/pose_camera", self.on_pose, 10)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.on_info, 10)
        self.timer = self.create_timer(0.25, self.tick)
        self._o3d_ready = False
        try:
            from robot_pose_pipeline.pointcloud import rgbd_to_pointcloud, voxel_downsample  # noqa: F401
            from robot_pose_pipeline.registration import transform_model_to_camera  # noqa: F401

            self._o3d_ready = True
        except Exception as exc:
            self.get_logger().warn(f"Open3D pipeline import failed ({exc}); node idle until fixed")

    def on_rgb(self, msg: Image):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._header = msg.header

    def on_depth(self, msg: Image):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self._header = msg.header

    def on_mask(self, msg: Image):
        self.mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def on_pose(self, msg: PoseStamped):
        from scipy.spatial.transform import Rotation

        q = msg.pose.orientation
        t = msg.pose.position
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        self.pose_T = T
        self._header = msg.header

    def tick(self):
        if not self._o3d_ready or self.rgb is None or self.depth is None or self.K is None:
            return
        from robot_pose_pipeline.pointcloud import open3d_to_numpy, rgbd_to_pointcloud, voxel_downsample
        from robot_pose_pipeline.pointcloud.convert import numpy_to_open3d
        from robot_pose_pipeline.registration import transform_model_to_camera

        scale = float(self.get_parameter("depth_scale_to_m").value)
        voxel = float(self.get_parameter("voxel_size").value)
        try:
            raw = rgbd_to_pointcloud(
                self.depth, self.K, rgb_bgr=self.rgb, depth_scale_to_m=scale, median_ksize=3
            )
            processed = voxel_downsample(raw, voxel)
            pts, _ = open3d_to_numpy(processed)
            self.pub_processed.publish(xyz_to_cloud(pts, self._header))
            if self.mask is not None:
                obj = rgbd_to_pointcloud(
                    self.depth,
                    self.K,
                    rgb_bgr=self.rgb,
                    mask=self.mask,
                    depth_scale_to_m=scale,
                    median_ksize=3,
                )
                opts, _ = open3d_to_numpy(obj)
                self.pub_object.publish(xyz_to_cloud(opts, self._header))
            if self.model_xyz is not None and self.pose_T is not None:
                cad = transform_model_to_camera(numpy_to_open3d(self.model_xyz), self.pose_T)
                cpts, _ = open3d_to_numpy(cad)
                self.pub_cad.publish(xyz_to_cloud(cpts, self._header))
        except Exception as exc:
            self.get_logger().warn(f"cloud process failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = CloudProcessor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
