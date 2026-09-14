from __future__ import annotations

import numpy as np


def matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Return ROS quaternion x,y,z,w from a 3x3 rotation matrix."""
    m = np.asarray(rotation, dtype=float)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def transform_message_to_matrix(transform) -> np.ndarray:
    result = np.eye(4)
    t, q = transform.translation, transform.rotation
    result[:3, :3] = quaternion_to_matrix(q.x, q.y, q.z, q.w)
    result[:3, 3] = [t.x, t.y, t.z]
    return result


def fill_pose_message(pose, transform: np.ndarray) -> None:
    pose.position.x, pose.position.y, pose.position.z = [float(v) for v in transform[:3, 3]]
    x, y, z, w = matrix_to_quaternion(transform[:3, :3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = x, y, z, w


def fill_transform_message(message, transform: np.ndarray) -> None:
    message.translation.x, message.translation.y, message.translation.z = [float(v) for v in transform[:3, 3]]
    x, y, z, w = matrix_to_quaternion(transform[:3, :3])
    message.rotation.x, message.rotation.y, message.rotation.z, message.rotation.w = x, y, z, w
