#!/usr/bin/env python3
"""Compose README GIF from synchronized RGB+RViz pairs (equal-width panels)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def imread_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def fit_contain(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def fit_cover(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (nw - tw) // 2)
    y0 = max(0, (nh - th) // 2)
    return resized[y0 : y0 + th, x0 : x0 + tw]


def focus_yellow(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
    ys, xs = np.where(mask > 0)
    if len(xs) < 200:
        return img
    cx, cy = int(xs.mean()), int(ys.mean())
    crop_w, crop_h = int(w * 0.78), int(h * 0.78)
    x0 = int(np.clip(cx - crop_w // 2, 0, max(0, w - crop_w)))
    y0 = int(np.clip(cy - crop_h // 2, 0, max(0, h - crop_h)))
    return img[y0 : y0 + crop_h, x0 : x0 + crop_w]


def crop_rviz_viewport(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    top = int(h * 0.10)
    bottom = int(h * 0.935)
    left = int(w * 0.28)
    sample = img[top:bottom, : max(1, int(w * 0.06))]
    if float(np.mean(sample)) < 45:
        left = int(w * 0.015)
    cropped = img[top:bottom, left:w]
    return cropped if cropped.size else img


def place(canvas: np.ndarray, img: np.ndarray, x: int, y: int, slot_w: int, slot_h: int) -> None:
    """Center img inside slot."""
    x0 = x + max(0, (slot_w - img.shape[1]) // 2)
    y0 = y + max(0, (slot_h - img.shape[0]) // 2)
    canvas[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img


def compose(left_bgr: np.ndarray, right_bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    canvas = np.full((out_h, out_w, 3), 24, dtype=np.uint8)
    pad = 8
    label_h = 26
    gap = 8
    # Equal panels
    panel_w = (out_w - 2 * pad - gap) // 2
    panel_h = out_h - 2 * pad - label_h

    # Same slot size + cover for both so panels look equal (no letterbox shrink).
    left = fit_cover(focus_yellow(left_bgr), panel_w, panel_h)
    right = fit_cover(crop_rviz_viewport(right_bgr), panel_w, panel_h)

    y = pad + label_h
    place(canvas, left, pad, y, panel_w, panel_h)
    place(canvas, right, pad + panel_w + gap, y, panel_w, panel_h)

    cv2.putText(canvas, "RGB (camera)", (pad, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "RViz result",
        (pad + panel_w + gap, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synced-dir", type=Path, required=True, help="Dir with rgb/ and rviz/ subfolders")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "assets" / "pipeline_demo.gif")
    args = parser.parse_args()

    rgb_files = sorted((args.synced_dir / "rgb").glob("*.png"))
    rviz_files = sorted((args.synced_dir / "rviz").glob("*.png"))
    n = min(len(rgb_files), len(rviz_files))
    if n < 2:
        raise SystemExit(f"need synced pairs under {args.synced_dir}/rgb and .../rviz")

    frames: list[Image.Image] = []
    for i in range(n):
        left = imread_bgr(rgb_files[i])
        right = imread_bgr(rviz_files[i])
        composed = compose(left, right, args.width, args.height)
        frames.append(Image.fromarray(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)))
        print(f"[{i+1}/{n}] {rgb_files[i].name} + {rviz_files[i].name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / args.fps))
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"wrote {args.output} ({len(frames)} frames, {args.output.stat().st_size/1024/1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
