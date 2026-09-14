from __future__ import annotations

import argparse
from pathlib import Path

from robot_pose_pipeline.synthetic_scene import write_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a tiny synthetic RGB-D + pose dataset for offline pipeline demos."
    )
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    manifest = write_synthetic_dataset(args.output_dir, frame_count=args.frames, seed=args.seed)
    print(f"wrote synthetic dataset -> {Path(args.output_dir).resolve()}")
    print(f"manifest -> {manifest.resolve()}")


if __name__ == "__main__":
    main()
