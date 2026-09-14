import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from .pose_utils import fill_pose_message, fill_transform_message, quaternion_to_matrix, transform_message_to_matrix


class PoseTransform(Node):
    def __init__(self):
        super().__init__("pose_transform")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("object_frame", "object_1")
        self.base_frame = self.get_parameter("base_frame").value
        self.object_frame = self.get_parameter("object_frame").value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.publisher = self.create_publisher(PoseStamped, "/perception/pose_base", 10)
        self.subscription = self.create_subscription(PoseStamped, "/perception/pose_camera", self.callback, 10)

    def callback(self, message: PoseStamped):
        try:
            tf = self.buffer.lookup_transform(self.base_frame, message.header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.2))
        except TransformException as exc:
            self.get_logger().warning(f"TF unavailable: {exc}")
            return
        camera_object = np.eye(4)
        q, p = message.pose.orientation, message.pose.position
        camera_object[:3, :3] = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        camera_object[:3, 3] = [p.x, p.y, p.z]
        base_object = transform_message_to_matrix(tf.transform) @ camera_object
        output = PoseStamped()
        output.header.stamp = message.header.stamp
        output.header.frame_id = self.base_frame
        fill_pose_message(output.pose, base_object)
        self.publisher.publish(output)
        object_tf = TransformStamped()
        object_tf.header = output.header
        object_tf.child_frame_id = self.object_frame
        fill_transform_message(object_tf.transform, base_object)
        self.broadcaster.sendTransform(object_tf)


def main(args=None):
    rclpy.init(args=args)
    node = PoseTransform()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
