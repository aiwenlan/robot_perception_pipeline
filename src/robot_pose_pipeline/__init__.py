"""Core geometry and evaluation tools for the robot pose project."""

from .mask_metrics import mask_coverage, mask_iou
from .pose_metrics import add_error, adds_error, evaluate_pose, projection_error_px, rotation_error_deg, translation_error
from .rgbd import depth_to_points
from .transforms import compose, invert_transform, make_transform, transform_points

__all__ = [
    "add_error",
    "adds_error",
    "compose",
    "depth_to_points",
    "evaluate_pose",
    "invert_transform",
    "make_transform",
    "mask_coverage",
    "mask_iou",
    "projection_error_px",
    "rotation_error_deg",
    "transform_points",
    "translation_error",
]
