# GraspNet setup (optional neural path)

本仓库 **默认抓取 demo 不依赖** 官方 GraspNet 神经网络。  
`scripts/demo_grasp_on_object_cloud.py` 使用 GraspNet 兼容的 `T_camera_grasp` 表示 + 几何候选，本机即可跑通。

若你之后要接 **graspnet-baseline 预训练推理**，按下面做（云 GPU / 本机 CUDA 均可）。

## A. 本仓默认路径（已实现）

```powershell
pip install -e ".[open3d]"
$env:PYTHONPATH="$PWD\src"
python scripts\demo_grasp_on_object_cloud.py
```

输出：

- `outputs/grasp_demo/grasps_overlay.png`
- `outputs/grasp_demo/grasps.npz`（`T_camera_grasp`, `width`, `score`, `depth`）
- `outputs/grasp_demo/T_camera_grasp_best.txt`

约定：`p_camera = T_camera_grasp · p_grasp`（相机光学系，米）。

## B. 官方 baseline（可选，不进主依赖）

1. 外部 clone（勿 vendoring 进本仓）：

```bash
git clone https://github.com/graspnet/graspnet-baseline.git
git clone https://github.com/graspnet/graspnetAPI.git
```

2. 按官方 README 安装依赖，下载 `checkpoint-rs.tar`（RealSense 更常迁移）。  
3. 用官方 `demo.py` 对 RGB-D 推理，导出 GraspGroup `(N,17)` `.npy`。  
4. 导入本仓可视化：

```powershell
python scripts\demo_grasp_on_object_cloud.py --grasps path\to\gg.npy
```

`grasps_from_graspnet_npy` 已支持 GraspGroup 17 列布局。

## 边界（面试必讲）

- 当前 **没有** 真实机械臂 / MoveIt2 / 闭环抓取。  
- 几何候选是最短可运行路径；NN GraspNet 是可选增强。  
- 权重与 GraspNet-1Billion 大数据保持 gitignore。
