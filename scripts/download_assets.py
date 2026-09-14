#!/usr/bin/env python3
"""Idempotently download FoundationPose weights, demo_data, and FastSAM-s.

demo_data sources (in order):
  1. HuggingFace candidate repos (if they ever host mustard0)
  2. Google Drive mustard0.zip (file id)
  3. Google Drive demo_data folder
  4. Optional local minisynth generator (YcbineoatReader-compatible stand-in)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "data" / "foundationpose" / "weights"
DEMO_DIR = ROOT / "data" / "foundationpose" / "demo_data"
MODELS_DIR = ROOT / "data" / "models"
DATASETS_DIR = ROOT / "data" / "datasets"
FASTSAM_PATH = MODELS_DIR / "FastSAM-s.pt"
MINISYNTH_DIR = DEMO_DIR / "mustard0_minisynth"

REFINER = "2023-10-28-18-33-37"
SCORER = "2024-01-11-20-02-45"
HF_REPO = "gpue/foundationpose-weights"
DEMO_GDRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing"
)
MUSTARD0_FILE_ID = "1AwV9sESDKMgXGUu2n1o0Pc4x2JGYdVB3"
FASTSAM_URL = (
    "https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.1.2/FastSAM-s.pt"
)
BOP_LM_REPO = "bop-benchmark/lm"

# HF datasets/models that *might* contain mustard0 (checked; most do not today).
HF_DEMO_CANDIDATES: list[tuple[str, str, str]] = [
    # (repo_id, repo_type, filename_or_empty_for_snapshot_scan)
    ("gpue/foundationpose-demo-data", "dataset", ""),
    ("gpue/FoundationPose-demo_data", "dataset", ""),
    ("wenbowen123/foundationpose-demo", "dataset", ""),
]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def ensure_pip_deps() -> None:
    _run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub", "gdown"])


def weights_ready() -> bool:
    for run in (REFINER, SCORER):
        if not (WEIGHTS_DIR / run / "model_best.pth").is_file():
            return False
    return True


def download_weights() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if weights_ready():
        print(f"[skip] FoundationPose weights already under {WEIGHTS_DIR}")
        return
    from huggingface_hub import snapshot_download

    print(f"[download] HF {HF_REPO} -> {WEIGHTS_DIR}")
    snapshot_download(repo_id=HF_REPO, local_dir=str(WEIGHTS_DIR))
    if not weights_ready():
        raise RuntimeError(
            f"Weights download finished but expected runs missing under {WEIGHTS_DIR}"
        )
    print("[ok] FoundationPose weights")


def _scene_ok(path: Path) -> bool:
    return path.is_dir() and (path / "rgb").is_dir() and any((path / "rgb").glob("*.png"))


def _find_mustard(root: Path) -> Path | None:
    for name in ("mustard0", "mustard0_minisynth"):
        direct = root / name
        if _scene_ok(direct):
            return direct
    for cand in root.rglob("mustard0"):
        if _scene_ok(cand):
            return cand
    return None


def official_demo_ready() -> bool:
    return _scene_ok(DEMO_DIR / "mustard0")


def any_demo_ready() -> bool:
    return _find_mustard(DEMO_DIR) is not None


def demo_ready() -> bool:
    # Prefer official; accept minisynth for pipeline continuity.
    return any_demo_ready()


def _unzip_if_needed(archive: Path, dest_dir: Path) -> None:
    print(f"[unzip] {archive} -> {dest_dir}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)


def _promote_mustard(staging: Path) -> None:
    mustard = None
    for cand in staging.rglob("mustard0"):
        if _scene_ok(cand):
            mustard = cand
            break
    if mustard is None and _scene_ok(DEMO_DIR / "mustard0"):
        return
    if mustard is None:
        raise RuntimeError(f"mustard0 scene not found under {staging}")
    dest = DEMO_DIR / "mustard0"
    if mustard.resolve() != dest.resolve():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(mustard), str(dest))


def _try_hf_demo_mirrors() -> bool:
    """Return True if official mustard0 was obtained from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files, repo_exists
    except ImportError:
        return False

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id, repo_type, preferred in HF_DEMO_CANDIDATES:
        try:
            if not repo_exists(repo_id, repo_type=repo_type):
                print(f"[skip] HF mirror missing: {repo_id}")
                continue
            files = list_repo_files(repo_id, repo_type=repo_type)
            mustard_files = [
                f
                for f in files
                if "mustard" in f.lower() and f.lower().endswith((".zip", ".tar", ".tar.gz"))
            ]
            if preferred and preferred in files:
                mustard_files = [preferred] + [f for f in mustard_files if f != preferred]
            if not mustard_files:
                # snapshot may already be extracted layout
                if any("mustard0/rgb" in f.replace("\\", "/") for f in files):
                    from huggingface_hub import snapshot_download

                    staging = DEMO_DIR / "_hf_staging"
                    if staging.exists():
                        shutil.rmtree(staging, ignore_errors=True)
                    print(f"[download] HF snapshot {repo_id} -> {staging}")
                    snapshot_download(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        local_dir=str(staging),
                    )
                    _promote_mustard(staging)
                    shutil.rmtree(staging, ignore_errors=True)
                    if official_demo_ready():
                        return True
                print(f"[skip] HF {repo_id} has no mustard archive")
                continue
            for fname in mustard_files[:3]:
                print(f"[download] HF {repo_id}/{fname}")
                path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    filename=fname,
                    local_dir=str(DEMO_DIR),
                )
                archive = Path(path)
                if archive.suffix.lower() == ".zip":
                    _unzip_if_needed(archive, DEMO_DIR)
                    _promote_mustard(DEMO_DIR)
                if official_demo_ready():
                    print(f"[ok] official mustard0 from HF {repo_id}")
                    return True
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] HF mirror {repo_id}: {exc}", file=sys.stderr)
    return False


def _try_gdown_mustard_zip() -> bool:
    zip_path = DEMO_DIR / "mustard0.zip"
    print(f"[download] gdown mustard0.zip id={MUSTARD0_FILE_ID}")
    try:
        _run(
            [
                sys.executable,
                "-m",
                "gdown",
                f"https://drive.google.com/uc?id={MUSTARD0_FILE_ID}",
                "-O",
                str(zip_path),
            ]
        )
    except subprocess.CalledProcessError as exc:
        print(f"[warn] mustard0.zip gdown failed: {exc}", file=sys.stderr)
        return False
    if zip_path.is_file() and zip_path.stat().st_size > 1_000_000:
        _unzip_if_needed(zip_path, DEMO_DIR)
        try:
            _promote_mustard(DEMO_DIR)
        except RuntimeError as exc:
            print(f"[warn] {exc}", file=sys.stderr)
            return False
        return official_demo_ready()
    return False


def _try_gdown_folder() -> bool:
    staging = DEMO_DIR / "_gdown_staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    print(f"[download] gdown folder -> {staging}")
    try:
        _run(
            [
                sys.executable,
                "-m",
                "gdown",
                DEMO_GDRIVE_FOLDER,
                "-O",
                str(staging),
            ]
        )
        for z in staging.rglob("*.zip"):
            _unzip_if_needed(z, staging)
        _promote_mustard(staging)
        shutil.rmtree(staging, ignore_errors=True)
        return official_demo_ready()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] folder download failed: {exc}", file=sys.stderr)
        shutil.rmtree(staging, ignore_errors=True)
        return False


def ensure_minisynth() -> Path:
    if _scene_ok(MINISYNTH_DIR):
        print(f"[skip] minisynth already at {MINISYNTH_DIR}")
        return MINISYNTH_DIR
    print(f"[generate] mustard0_minisynth -> {MINISYNTH_DIR}")
    _run([sys.executable, str(ROOT / "scripts" / "make_mustard0_minisynth.py"), "--out", str(MINISYNTH_DIR)])
    if not _scene_ok(MINISYNTH_DIR):
        raise RuntimeError("minisynth generation failed")
    return MINISYNTH_DIR


def download_demo_data(allow_minisynth: bool = True) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    if official_demo_ready():
        print(f"[skip] official mustard0 already under {DEMO_DIR}")
        return

    print("[info] Trying HF mirrors first, then Google Drive…")
    if _try_hf_demo_mirrors():
        return
    if _try_gdown_mustard_zip():
        print(f"[ok] demo_data at {DEMO_DIR / 'mustard0'}")
        return
    if _try_gdown_folder():
        print(f"[ok] demo_data at {DEMO_DIR / 'mustard0'}")
        return

    manual = DEMO_DIR / "README_MANUAL.md"
    msg = (
        "Official FoundationPose mustard0 unavailable (Drive quota / no HF mirror). "
        f"See {manual}"
    )
    if allow_minisynth:
        ensure_minisynth()
        print(f"[warn] {msg}")
        print(f"[ok] using stand-in scene {MINISYNTH_DIR}")
        return
    raise RuntimeError(msg)


def download_fastsam() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if FASTSAM_PATH.is_file() and FASTSAM_PATH.stat().st_size > 1_000_000:
        print(f"[skip] FastSAM already at {FASTSAM_PATH}")
        return
    print(f"[download] FastSAM-s.pt -> {FASTSAM_PATH}")
    import urllib.request

    urllib.request.urlretrieve(FASTSAM_URL, FASTSAM_PATH)
    if not FASTSAM_PATH.is_file():
        raise RuntimeError("FastSAM download failed")
    print("[ok] FastSAM-s.pt")


def download_linemod_models(optional: bool = True) -> None:
    out = DATASETS_DIR / "linemod"
    marker = out / "models"
    if marker.is_dir() and any(marker.iterdir()):
        print(f"[skip] LINEMOD models already under {marker}")
        return
    out.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        print(f"[download] HF {BOP_LM_REPO} lm_models.zip -> {out}")
        path = hf_hub_download(
            repo_id=BOP_LM_REPO,
            repo_type="dataset",
            filename="lm_models.zip",
            local_dir=str(out),
        )
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out)
        print(f"[ok] LINEMOD models under {out}")
    except Exception as exc:  # noqa: BLE001
        msg = f"optional LINEMOD models download failed: {exc}"
        if optional:
            print(f"[warn] {msg}", file=sys.stderr)
            return
        raise RuntimeError(msg) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--skip-fastsam", action="store_true")
    parser.add_argument(
        "--no-minisynth",
        action="store_true",
        help="Fail if official mustard0 cannot be downloaded (do not generate stand-in)",
    )
    parser.add_argument(
        "--with-linemod-models",
        action="store_true",
        help="Also fetch BOP LINEMOD object meshes (small; no full RGB-D)",
    )
    parser.add_argument("--no-pip", action="store_true", help="Do not pip install deps")
    args = parser.parse_args()

    if not args.no_pip:
        ensure_pip_deps()

    errors: list[str] = []
    if not args.skip_weights:
        try:
            download_weights()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"weights: {exc}")
            print(f"[FAIL] weights: {exc}", file=sys.stderr)
    if not args.skip_demo:
        try:
            download_demo_data(allow_minisynth=not args.no_minisynth)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"demo_data: {exc}")
            print(f"[FAIL] demo_data: {exc}", file=sys.stderr)
    if not args.skip_fastsam:
        try:
            download_fastsam()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"fastsam: {exc}")
            print(f"[FAIL] fastsam: {exc}", file=sys.stderr)
    if args.with_linemod_models:
        try:
            download_linemod_models(optional=False)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"linemod: {exc}")
            print(f"[FAIL] linemod: {exc}", file=sys.stderr)
    else:
        download_linemod_models(optional=True)

    print()
    print("Summary:")
    print(f"  weights ready: {weights_ready()} ({WEIGHTS_DIR})")
    print(f"  official mustard0: {official_demo_ready()} ({DEMO_DIR / 'mustard0'})")
    print(f"  any demo scene: {any_demo_ready()} ({_find_mustard(DEMO_DIR)})")
    print(f"  FastSAM ready: {FASTSAM_PATH.is_file()} ({FASTSAM_PATH})")
    print(f"  LINEMOD models: {(DATASETS_DIR / 'linemod' / 'models').exists()}")
    if errors:
        print(f"{len(errors)} step(s) failed.")
        return 1
    print("All requested assets ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
