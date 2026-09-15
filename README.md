# RGB-D Robot 6D Pose Perception Pipeline

![Pipeline demo](docs/assets/pipeline_demo.gif)

可验证的工程链路：公开 RGB-D → 掩膜 → FoundationPose 6D 位姿 → 手眼/坐标变换 → ROS2 离线回放。

```text
Camera calib → Detector bbox / FastSAM mask → FoundationPose (T_camera_object)
→ Hand-eye AX=XB → T_base_object = T_base_camera @ T_camera_object
→ ROS2 TF2 / PointCloud2 / RViz2 / rosbag2
```

**本仓库不包含：** 完整 PCL 工程化、FK/IK、MoveIt2、真实机械臂联调。  
Open3D 点云/ICP/TSDF 为可选扩展；PCL 仅概念对照，见 [`docs/PCL_EQUIVALENTS.md`](docs/PCL_EQUIVALENTS.md)。

## Highlights

| Item | Result |
|---|---|
| Chessboard calibration | mean reprojection ≈ **0.14 px** |
| FastSAM vs GT-mask pose (mustard0 frame) | ≈ **1.23° / 0.52 mm** (IoU ≈ 0.95) |
| YCB-V vs dataset GT (50 frames, mean) | rot 1.54°; trans 2.45 mm; **ADD-S 1.62 mm** |
| Hand-eye | noise ↑ → error ↑; more poses → error ↓ (**simulated** extrinsics) |
| ROS2 | Image / Depth / CameraInfo / cloud / pose / TF; dual entry (live + bag) |

Compact metrics: [`docs/results_summary.json`](docs/results_summary.json). Full experiment report: [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).

## Architecture

```mermaid
flowchart LR
  D[Public RGB-D / synthetic] --> RGB[RGB + bbox]
  D --> DEP[Depth + CameraInfo]
  RGB --> GT[GT mask]
  RGB --> FS[FastSAM mask]
  GT --> FP[FoundationPose]
  FS --> FP
  DEP --> FP
  CAD[CAD / mesh] --> FP
  FP --> CP[T_camera_object]
  HE[Hand-eye AX=XB] --> BC[T_base_camera]
  CP --> PT[T_base_object]
  BC --> PT
  PT --> ROS[ROS2 TF2 + PointCloud2 + RViz2 + rosbag2]
```

## Quick start

```powershell
# from repo root
conda activate <your-env>
$env:PYTHONPATH="$PWD\src"

python -m unittest discover -s tests -v
python scripts\run_core_demo.py

python scripts\calibrate_camera.py "data/calibration/opencv_left/*.jpg" --cols 7 --rows 6 --square-size 0.03
python scripts\undistort_preview.py --calibration outputs\camera_calibration.json --image data\calibration\opencv_left\left01.jpg

# optional: download large assets into data/ (gitignored)
python scripts\download_assets.py
python scripts\link_foundationpose_assets.py
python scripts\check_resources.py
```

Regenerate the README GIF (synced RGB+RViz pairs):

```powershell
# 1) In WSL: launch offline pipeline + rviz2, then capture synced pairs
# 2) On Windows:
python scripts\make_readme_gif.py `
  --synced-dir outputs\ros2_acceptance\synced_gif `
  --fps 5 --width 1000 --height 480
```

## ROS2 offline replay (WSL2 + Jazzy)

See [`docs/ROS2_RUNBOOK.md`](docs/ROS2_RUNBOOK.md).

```bash
# from repo root on WSL
source /opt/ros/jazzy/setup.bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash

ros2 launch robot_pose_pipeline_ros offline_pipeline.launch.py \
  manifest:=$(pwd)/../data/synthetic/manifest.jsonl \
  base_camera_pose:=$(pwd)/../data/synthetic/T_base_camera.txt

rviz2 -d src/robot_pose_pipeline_ros/rviz/offline_pipeline.rviz
```

FoundationPose on cloud GPU: [`docs/CLOUD_FOUNDATIONPOSE.md`](docs/CLOUD_FOUNDATIONPOSE.md).

## Open3D 3D vision extensions

主链路之外的可选模块（不破坏已有 FP / ROS 回放）：深度预处理 → 点云生成/滤波/平面/聚类 → CAD+ICP → ORB+PnP 支路 → TSDF → 可选 ROS 点云话题。

```powershell
pip install -e ".[open3d]"
$env:PYTHONPATH="$PWD\src"
python scripts\demo_depth_pointcloud.py
python scripts\demo_filter_segment_cluster.py
python scripts\demo_icp_cad_align.py
python scripts\demo_orb_pnp.py
python scripts\demo_tsdf_fusion.py
python scripts\demo_ros_cloud_topics.py
# regenerate README previews:
python scripts\make_readme_viz.py
```

```text
RGB-D → depth prep → Open3D cloud → filter / plane / cluster
→ mask → FoundationPose → CAD @ T_camera_object → ICP
→ T_base_object → ROS2 (raw / object / CAD clouds + TF)
→ optional TSDF

Experimental: RGB → ORB → PnP → compare with FoundationPose
```

### Visual results (synthetic demo)

**Depth → scene / object point cloud**

![M1 point cloud](docs/assets/viz_m1_pointcloud.png)

**Filtered cluster**

![M2 cluster](docs/assets/viz_m2_cluster.png)

**CAD model alignment — ICP before / after**

![M3 ICP](docs/assets/viz_m3_icp.png)

**ORB feature matches (query | train)**

![M4 ORB](docs/assets/viz_m4_orb_matches.png)

**Multi-frame TSDF mesh**

![M5 TSDF](docs/assets/viz_m5_tsdf.png)

**ROS-oriented cloud topics: raw / processed / object / CAD-aligned**

![M6 clouds](docs/assets/viz_m6_clouds.png)

Open3D ↔ PCL 接口对照：[`docs/PCL_EQUIVALENTS.md`](docs/PCL_EQUIVALENTS.md)。

## PCL mini demo (C++)

独立工程 [`pcl_demo/`](pcl_demo/)（不改写 Python/ROS 主链路）：

```text
PCD → VoxelGrid → PassThrough/SOR → RANSAC plane → Euclidean clustering → ICP
```

```bash
# WSL Ubuntu 24.04 + libpcl-dev
cd pcl_demo
python3 scripts/make_demo_clouds.py
mkdir -p build && cd build
cmake .. -Wno-dev && cmake --build . -j
./pcl_mini_demo --input ../data/scene.pcd --model ../data/model.pcd --out ../outputs
```

**PCL pipeline stages (input → no-plane → object cluster → ICP aligned)**

![PCL mini demo](docs/assets/viz_pcl_demo.png)

详见 [`pcl_demo/README.md`](pcl_demo/README.md)。

## Repository layout

```text
robot_perception_pipeline/
├─ pcl_demo/                  # standalone PCL C++ mini pipeline (WSL)
├─ src/robot_pose_pipeline/   # calib, transforms, hand-eye, metrics, RGB-D,
│                             # depth/, pointcloud/, registration/, matching/, reconstruction/
├─ scripts/                   # CLI + Open3D demos + README viz helpers
├─ ros2_ws/                   # ROS2 offline replay (+ optional cloud_processor)
├─ data/
│  ├─ calibration/            # chessboard samples (tracked)
│  ├─ synthetic/              # tiny synthetic demo (tracked)
│  ├─ models/                 # FastSAM weights (gitignored)
│  ├─ foundationpose/         # FP weights + demo_data (gitignored)
│  └─ datasets/               # optional YCB/BOP subsets (gitignored)
├─ outputs/                   # local run artifacts (gitignored)
├─ tests/
└─ docs/
```

## Documentation

| Doc | Purpose |
|---|---|
| [DOWNLOAD_CHECKLIST.md](docs/DOWNLOAD_CHECKLIST.md) | Asset download checklist |
| [ENVIRONMENT.md](docs/ENVIRONMENT.md) | Windows / WSL2 / Docker / ROS2 |
| [CLOUD_FOUNDATIONPOSE.md](docs/CLOUD_FOUNDATIONPOSE.md) | Cloud GPU FoundationPose notes |
| [RESOURCES.md](docs/RESOURCES.md) | External resources |
| [DATA_AND_CONVENTIONS.md](docs/DATA_AND_CONVENTIONS.md) | Frames & pose convention |
| [ROS2_RUNBOOK.md](docs/ROS2_RUNBOOK.md) | Build / launch / bag / RViz |
| [PCL_EQUIVALENTS.md](docs/PCL_EQUIVALENTS.md) | Open3D ↔ PCL mapping |
| [PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | Experiment report |
| [results_summary.json](docs/results_summary.json) | Compact metrics |

## Known limitations

1. Hand-eye / `T_base_camera` are **simulated**, not real-robot calibration.  
2. Official mustard0 demo has **no annotated GT poses**; GT metrics use YCB-V.  
3. FastSAM experiment bbox is derived from GT mask (not an independent detector).  
4. FoundationPose was validated primarily on a **cloud RTX 4090**; laptop 6GB GPU is optional.  
5. Large bags, screenshots, and demo videos stay local under `outputs/` (gitignored).  
6. Synthetic demos are for algorithm learning; weak texture may make ORB+PnP fall back to a synthetic PnP check.

## Acceptance checklist

1. Camera calibration JSON + undistort preview + reprojection error  
2. Hand-eye `AX=XB` (+ noise sweep)  
3. FoundationPose: GT mask → `T_camera_object`  
4. FastSAM vs GT-mask pose comparison  
5. `T_base_object = T_base_camera @ T_camera_object`  
6. ROS2 launch: Image / CameraInfo / PointCloud2 / Pose / TF / rosbag2 / RViz2  
7. (Optional) Open3D demos produce PLY/PNG under `outputs/` and README viz assets  
8. (Optional) `pcl_demo/` builds on WSL and writes staged PCD + ICP fitness  
