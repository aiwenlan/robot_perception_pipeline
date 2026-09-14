from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach FoundationPose ob_in_cam results to a JSONL manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pose-dir", required=True, help="FoundationPose debug/ob_in_cam directory")
    parser.add_argument("--output", default="data/manifest_with_pose.jsonl")
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    pose_dir = Path(args.pose_dir).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    attached = 0
    lines = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        record = json.loads(line)
        pose = pose_dir / f"{record['frame_id']}.txt"
        if pose.is_file():
            record["predicted_pose"] = str(pose)
            attached += 1
        lines.append(json.dumps(record, ensure_ascii=False))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"attached {attached}/{len(lines)} poses -> {output.resolve()}")


if __name__ == "__main__":
    main()
