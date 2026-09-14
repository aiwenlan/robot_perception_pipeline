#!/usr/bin/env python3
"""Create Windows junctions from course FoundationPose-main to local asset copies."""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FP = ROOT.parent / "code" / "FoundationPose-main"
LOCAL_WEIGHTS = ROOT / "data" / "foundationpose" / "weights"
LOCAL_DEMO = ROOT / "data" / "foundationpose" / "demo_data"
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF


def is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    except Exception:
        return False
    if attrs == INVALID_FILE_ATTRIBUTES:
        return False
    return bool(attrs & FILE_ATTRIBUTE_REPARSE_POINT)


def remove_link(path: Path) -> None:
    """Remove a junction/symlink without deleting target contents."""
    if path.is_symlink():
        path.unlink(missing_ok=True)
        return
    # Prefer cmd rmdir for junctions (including dangling ones).
    completed = subprocess.run(
        ["cmd", "/c", "rmdir", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode == 0 or not path.exists():
        return
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f'Remove-Item -LiteralPath "{path}" -Force',
        ],
        check=False,
    )


def same_destination(link: Path, target: Path) -> bool:
    """True if link already resolves to target (junction or identical path)."""
    try:
        if not link.exists():
            return False
        return link.resolve() == target.resolve()
    except OSError:
        return False


def make_junction(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.is_dir():
        raise RuntimeError(f"Target missing: {target}")
    link.parent.mkdir(parents=True, exist_ok=True)

    if same_destination(link, target):
        print(f"[skip] already linked: {link} -> {target}")
        return

    if is_reparse_point(link):
        print(f"[replace] existing reparse point {link}")
        remove_link(link)
    elif link.exists():
        if link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        elif link.is_dir():
            raise RuntimeError(
                f"{link} is a real non-empty directory that does not point to {target}. "
                "Move/rename it, then re-run this script."
            )
        else:
            link.unlink()

    ps = (
        f'$null = New-Item -ItemType Junction -Path "{link}" -Target "{target}"; '
        f'"OK"'
    )
    print(f"+ New-Item Junction {link.name}", flush=True)
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not same_destination(link, target):
        err = (completed.stdout or "") + "\n" + (completed.stderr or "")
        sys.stderr.buffer.write(err.encode("utf-8", errors="replace") + b"\n")
        raise RuntimeError(f"junction failed for {link} -> {target}")
    print(f"[ok] junction: {link.name} -> {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundationpose",
        type=Path,
        default=DEFAULT_FP,
        help="Path to FoundationPose-main repository",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fp = args.foundationpose.resolve()
    if not (fp / "estimater.py").is_file():
        raise SystemExit(f"FoundationPose repo not found: {fp}")

    pairs = [
        (fp / "weights", LOCAL_WEIGHTS),
        (fp / "demo_data", LOCAL_DEMO),
    ]
    errors = 0
    for link, target in pairs:
        print(f"plan: {link}  =>  {target}")
        if args.dry_run:
            continue
        try:
            make_junction(link, target)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[FAIL] {exc}", file=sys.stderr)

    print()
    if errors:
        print(f"{errors} junction(s) failed.")
        return 1
    print("Done. Official scripts can use FoundationPose-main/weights and demo_data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
