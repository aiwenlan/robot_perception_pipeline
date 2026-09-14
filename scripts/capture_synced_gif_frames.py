#!/usr/bin/env python3
"""Capture synchronized RGB + RViz frame pairs while the offline pipeline plays.

Run inside WSL with ROS 2 sourced, after launch + rviz2 are up:

  python3 scripts/capture_synced_gif_frames.py --out /tmp/synced --max-frames 51
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class SyncedCapture(Node):
    def __init__(self, out_dir: Path, rviz_wid: str, max_frames: int, settle_s: float):
        super().__init__("synced_gif_capture")
        self.out_dir = out_dir
        self.rgb_dir = out_dir / "rgb"
        self.rviz_dir = out_dir / "rviz"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.rviz_dir.mkdir(parents=True, exist_ok=True)
        self.wid = rviz_wid
        self.max_frames = max_frames
        self.settle_s = settle_s
        self.bridge = CvBridge()
        self.count = 0
        self.last_stamp = None
        self.done = False
        self.create_subscription(Image, "/camera/color/image_raw", self.on_image, 10)

    def on_image(self, msg: Image) -> None:
        if self.done:
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self.last_stamp:
            return
        self.last_stamp = stamp
        # Let RViz catch up with TF/cloud for this frame.
        time.sleep(self.settle_s)

        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        idx = f"{self.count + 1:03d}"
        rgb_path = self.rgb_dir / f"{idx}.png"
        rviz_path = self.rviz_dir / f"{idx}.png"
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return
        buf.tofile(str(rgb_path))

        subprocess.run(
            ["import", "-window", self.wid, str(rviz_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.count += 1
        self.get_logger().info(f"captured pair {idx}")
        if self.count >= self.max_frames:
            self.done = True


def find_rviz_wid() -> str:
    out = subprocess.check_output(["bash", "-lc", "xwininfo -root -tree"], text=True)
    best_wid = ""
    best_area = 0
    for line in out.splitlines():
        if "rviz2" not in line and "RViz" not in line:
            continue
        # 0xabc...  WxH+X+Y
        parts = line.strip().split()
        if not parts:
            continue
        wid = parts[0]
        for token in parts:
            if "x" in token and "+" in token:
                try:
                    wh = token.split("+", 1)[0]
                    w_s, h_s = wh.split("x")
                    area = int(w_s) * int(h_s)
                    if area > best_area:
                        best_area = area
                        best_wid = wid
                except ValueError:
                    pass
                break
    if not best_wid:
        raise RuntimeError("RViz window not found")
    return best_wid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=51)
    parser.add_argument("--settle", type=float, default=0.12)
    parser.add_argument("--idle-timeout", type=float, default=8.0, help="Exit if no new frames for this many seconds")
    args = parser.parse_args()

    wid = find_rviz_wid()
    print(f"RViz WID={wid}")

    rclpy.init()
    node = SyncedCapture(args.out, wid, args.max_frames, args.settle)
    last_count = -1
    last_progress = time.time()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.count != last_count:
                last_count = node.count
                last_progress = time.time()
            elif node.count > 0 and (time.time() - last_progress) > args.idle_timeout:
                print(f"idle timeout after {node.count} pairs")
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f"done pairs={node.count} -> {args.out}")
    return 0 if node.count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
