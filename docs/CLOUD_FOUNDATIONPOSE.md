# Cloud FoundationPose runbook

Notes for running FoundationPose on a **cloud NVIDIA GPU** (validated on RTX 4090, Ubuntu + Docker). Local development covers calibration, hand-eye, metrics, and ROS2; the cloud instance is used only for FoundationPose inference when local VRAM is limited.

---

## 1. Roles

| Side | Responsibility |
|---|---|
| Cloud GPU | Docker GPU FoundationPose (`run_demo.py` / registration) |
| Local machine | Calibration, FastSAM, metrics, ROS2 offline replay |

Typical cloud workspace:

```text
~/workspace/FoundationPose
```

Local assets (gitignored in this repo):

```text
data/foundationpose/weights/
data/foundationpose/demo_data/
```

---

## 2. Environment

1. Install Docker + NVIDIA Container Toolkit; verify `nvidia-smi` inside a GPU container.  
2. Upload or sync FoundationPose **weights** and demo data (e.g. official `mustard0`).  
3. Clone / mount the FoundationPose source; symlink weights and `demo_data` to the paths expected by `run_demo.py`.

### Pulling the official image

Official image (large, ~32GB): `wenbowen123/foundationpose`.

If the cloud host cannot reach Docker Hub directly, use a temporary **SSH reverse tunnel** from a machine that can pull:

```powershell
ssh -N -R 12450:127.0.0.1:12450 ubuntu@<CLOUD_PUBLIC_IP>
```

Point Docker on the cloud host at that local proxy port, `docker pull`, then remove the proxy when finished.

---

## 3. RTX 4090 / sm_89 caveat

The public image ships an older CUDA toolkit (~11.3). Building nvdiffrast for Ada (sm_89) may fail with:

```text
nvcc fatal: Unsupported gpu architecture 'compute_89'
```

**Fix:** mount a newer host CUDA toolkit into the container and set arch flags, for example:

```bash
sudo docker run -d --gpus all --name foundationpose \
  -v "$HOME/workspace:/home/ubuntu/workspace" \
  -v /usr/local/cuda-12.8:/usr/local/cuda-host:ro \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e TORCH_CUDA_ARCH_LIST=8.9 \
  -e CUDA_HOME=/usr/local/cuda-host \
  -e PATH=/usr/local/cuda-host/bin:$PATH \
  -e LD_LIBRARY_PATH=/usr/local/cuda-host/lib64:$LD_LIBRARY_PATH \
  foundationpose:latest sleep infinity
```

Trigger nvdiffrast rebuild via `RasterizeCudaContext()` inside the container. Headless hosts: install `xvfb` and run with `xvfb-run -a python run_demo.py ...`.

---

## 4. Runs completed in this project

### Official mustard0 sequence

```bash
cd ~/workspace/FoundationPose
xvfb-run -a python run_demo.py --debug 2
```

- ~**737** frames of `T_camera_object` under `debug/ob_in_cam/*.txt`  
- Synced locally to `outputs/foundationpose_cloud/mustard0/` (gitignored)

### FastSAM mask vs GT mask (single frame)

Same frame registration with GT mask vs FastSAM mask. Summary in [`results_summary.json`](results_summary.json) (`mustard0_fastsam_vs_gt_mask`). Reference pose is **GT-mask registration** (mustard0 has no annotated GT poses).

### YCB-V BOP subset (GT metrics)

- Scene `000050`, object id `5`, 50 frames  
- Bake BOP `depth_scale` into depth before FoundationPose readers that assume mm→m via `/1e3` only  
- BOP `mask_visib` names use `{im_id}_{gt_list_index}.png` (index ≠ obj_id)  
- Mean ADD-S ≈ **1.62 mm** — see [`results_summary.json`](results_summary.json)

Local helpers: `scripts/convert_bop_ycbv_scene.py`, `scripts/evaluate_pose_batch.py`.

---

## 5. Cost hygiene

- Start the instance only for FoundationPose jobs; shut down when idle.  
- Keep the data disk if you need to avoid re-pulling the 32GB image; confirm local `outputs/` are synced before releasing storage.

---

## 6. Attach poses back into a manifest

```powershell
python scripts\attach_pose_results.py `
  --manifest data\foundationpose\demo_data\mustard0_manifest.jsonl `
  --pose-dir outputs\foundationpose_cloud\mustard0\ob_in_cam `
  --output data\foundationpose\demo_data\mustard0_ros_manifest.jsonl
```

Prefer **relative** pose paths inside JSONL for portability.
