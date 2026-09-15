# PCL Mini Demo

独立 C++ 小工程（**不**改写仓库 Python / ROS 主链路）。用于证明：Open3D 主线之外，也有 PCL 实操。

## Pipeline

```text
PCD input
→ VoxelGrid
→ PassThrough(Z) + StatisticalOutlierRemoval
→ RANSAC plane removal (only if plane is dominant)
→ EuclideanClusterExtraction
→ ICP (model/source → largest object cluster/target)
```

坐标系：点云默认在 **camera optical frame**，单位 **米**。  
ICP：`T_target_source`，即 `p_target ≈ T_target_source * p_source`。

## Dependencies (WSL Ubuntu 24.04)

```bash
sudo apt update
sudo apt install -y build-essential cmake libpcl-dev
```

本仓库开发机已验证：PCL 1.14 + `libpcl-dev`。

## Build & run

```bash
# from repo root on WSL, e.g. /mnt/d/.../robot_perception_pipeline
cd pcl_demo
python3 scripts/make_demo_clouds.py          # writes data/scene.pcd + data/model.pcd
mkdir -p build && cd build
cmake ..
cmake --build . -j"$(nproc)"
./pcl_mini_demo \
  --input ../data/scene.pcd \
  --model ../data/model.pcd \
  --out ../outputs
```

Windows 可先在 WSL 编译运行；原生 MSVC+PCL 不作为本 demo 验收路径。

## Outputs

| File | Meaning |
|---|---|
| `00_input.pcd` | 原始输入 |
| `01_voxel.pcd` | 体素下采样 |
| `02_passthrough.pcd` | Z 裁剪 |
| `03_sor.pcd` | 统计离群点去除 |
| `04_no_plane.pcd` | 去平面后 |
| `05_cluster_*.pcd` | 各欧氏聚类 |
| `06_object_cluster.pcd` | 最大簇（观测目标） |
| `07_model_source.pcd` | ICP source（CAD/扰动模型） |
| `08_model_after_icp.pcd` | ICP 对齐后 |

终端会打印每步点数、平面方程、cluster AABB、ICP fitness。

## Open3D ↔ PCL

对照表见仓库 [`docs/PCL_EQUIVALENTS.md`](../docs/PCL_EQUIVALENTS.md)。

| 本 demo 步骤 | PCL | Open3D（本仓已有） |
|---|---|---|
| Voxel | `pcl::VoxelGrid` | `voxel_downsample` |
| Z crop | `pcl::PassThrough` | `crop_box` |
| SOR | `pcl::StatisticalOutlierRemoval` | `statistical_outlier_removal` |
| Plane | `pcl::SACSegmentation` | `segment_plane` |
| Cluster | `pcl::EuclideanClusterExtraction` | `cluster_dbscan`（近似） |
| ICP | `pcl::IterativeClosestPoint` | `icp_point_to_point` |
