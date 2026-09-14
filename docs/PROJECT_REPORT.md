# 机器人感知 Pipeline 实验报告

## 1. 目标与范围

**目标：** 打通可量化验证的 RGB-D → 6D 位姿 → 手眼/坐标变换 → ROS2 离线回放链路。

**主数据：**
- FoundationPose 官方 `mustard0` demo（全序列 tracking）
- BOP **YCB-V** 测试子集：scene `000050`、物体 id `5`（mustard bottle），50 帧，相对 **数据集 GT pose** 评估

**本周完成：** 相机标定样例、合成手眼与噪声实验、FastSAM vs GT-mask 位姿对比、云端 FoundationPose、YCB GT 指标表、WSL2 ROS2 Jazzy 双入口回放。

**明确排除：** PCL 滤波/平面/聚类、FK/IK、MoveIt2、真实机械臂联调；学习式路线（RT-1/ACT/LeRobot）本阶段不做。

## 2. 环境

| 项 | 配置 |
|---|---|
| Windows | 开发 / conda 或 venv：核心几何与评估 |
| WSL2 | Ubuntu 24.04 + **ROS2 Jazzy desktop** |
| 云端 | 云 GPU（RTX 4090）+ Docker `foundationpose`；主机 CUDA≥12 挂载以支持 sm_89（见 `docs/CLOUD_FOUNDATIONPOSE.md`） |
| 本机 GPU | 可选笔记本 GPU（如 RTX 3060 6GB）；本项目 FP 主验证在云端 |
| 权重 | `data/foundationpose/weights/`（refiner / scorer，gitignored） |

## 3. 坐标系与位姿定义

约定：`T_parent_child` 把 **child 坐标系下的点** 变到 **parent**。

```text
T_base_object = T_base_camera @ T_camera_object
```

| Frame | 含义 |
|---|---|
| `base_link` | 机器人基座（本项目手眼外参为**模拟**） |
| `camera_color_optical_frame` | 相机光学系（ROS：x 右、y 下、z 前） |
| object | 物体 CAD / mesh 系 |

云端 / 评估输出的 `ob_in_cam/*.txt` 即 `T_camera_object`（4×4）。  
ROS 中由 `static_camera_tf` 发布模拟 `T_base_camera`，`pose_transform` 得到 `/perception/pose_base`。

**重要表述：** `T_base_camera` / 手眼结果均为仿真或演示外参，**不是**真实机器人标定结果。YCB / mustard0 自带内参 ≠ 棋盘格标定得到的 `K`。

## 4. 相机标定

- 数据：OpenCV 公开棋盘格样例 `data/calibration/opencv_left/`
- 规格：7×6 内角点，方格 0.03 m（脚本参数）
- 结果：`outputs/camera_calibration.json`
  - RMS ≈ **0.156 px**
  - 平均重投影误差 ≈ **0.138 px**
- 另有角点预览与去畸变图：`outputs/calibration_previews/`、`outputs/undistort_preview.png`

## 5. 手眼标定

- 类型：Eye-in-Hand，合成姿态对 `AX=XB`
- 方法：Tsai / Park / Horaud（及既有多方法对比）
- 噪声实验：`scripts/hand_eye_noise_sweep.py` → `outputs/hand_eye_noise_sweep.json`

摘要（模拟数据）：
- 无噪声：旋转/平移误差 ≈ 0
- 噪声增大（如 2° / 数毫米）：误差同步上升
- 同样噪声下，姿态对数 8→50，误差整体下降
- 本实验中 Park / Horaud 略稳于 Tsai

## 6. 6D 位姿估计

### 6.1 mustard0（官方 demo）

- 云端 `run_demo.py`：GT mask 首帧 register + 全序列 tracking，**737** 帧 `T_camera_object`
- 产物：`outputs/foundationpose_cloud/mustard0/ob_in_cam/`

### 6.2 FastSAM vs GT-mask（同帧对比）

帧 `1581120424100262102`：bbox 由 GT mask 推导（报告注明：非独立检测器）。

| 项 | 数值 |
|---|---:|
| Mask IoU | 0.945 |
| 相对 GT-mask 位姿：旋转 / 平移 | 1.23° / 0.52 mm |
| ADD / ADD-S / 投影 | 0.88 mm / 0.77 mm / 0.32 px |

参考为 **GT-mask 跑出的位姿**，不是 mustard0 annotated GT（demo 无 annotated poses）。

### 6.3 YCB-V BOP 子集（相对数据集 GT）

- Scene `000050`，obj_id `5`，50 帧；GT mask 每帧 `register`
- 深度按 BOP `depth_scale`（0.1）烘焙后供 FoundationPose 读取
- 指标：`outputs/ycbv_mustard_000050/pose_metrics_vs_gt.json`

## 7. 定量结果

### YCB-V vs GT（主表）

| Mask | Rot err (deg) | Trans err (m) | ADD (m) | ADD-S (m) | Proj err (px) |
|---|---:|---:|---:|---:|---:|
| GT (mean, 50 frames) | 1.54 | 0.00245 | 0.00271 | **0.00162** | 3.49 |
| GT (median) | 1.55 | 0.00242 | 0.00263 | 0.00160 | 3.46 |

### mustard0 FastSAM vs GT-mask 位姿差

| Mask | Rot err (deg) | Trans err (m) | ADD | ADD-S | Proj err (px) |
|---|---:|---:|---:|---:|---:|
| FastSAM（相对 GT-mask pose） | 1.23 | 0.00052 | 0.00088 | 0.00077 | 0.32 |

## 8. ROS2 验证

- 包：`ros2_ws/src/robot_pose_pipeline_ros`
- Launch：`offline_pipeline.launch.py`（`dataset_player` / `static_camera_tf` / `pose_transform`）
- 话题：Image、Depth、CameraInfo、mask、PointCloud2、pose_camera、pose_base、`/tf`、`/tf_static`
- 数据入口：`data/foundationpose/demo_data/mustard0_ros_manifest.jsonl` + 模拟 `T_base_camera_demo.txt`
- **入口 1（live launch）+ 入口 2（bag play）** 已在本地验收；产物目录 `outputs/ros2_acceptance/`（gitignored）
- 冒烟 bag 示例：`outputs/ros2_smoke/bag_mustard0/`（本地）

**RViz2：** 使用预置配置 `ros2_ws/src/robot_pose_pipeline_ros/rviz/offline_pipeline.rviz`（Fixed Frame=`base_link`，TF / PointCloud2 / Pose / Image）。截图与演示视频保存在本地 `outputs/`，不纳入 Git。

## 9. 失败样例与注意点

1. **深度单位：** BOP `depth_scale` 未烘焙时，平移可偏差约 10×；已在转换脚本中修复。  
2. **mask 命名：** BOP `mask_visib` 第二段是 `scene_gt` 列表下标，不是 obj_id。  
3. **mustard0：** 无 annotated GT pose，只能做 FastSAM vs GT-mask 互比或可视化。  
4. **ROS optical frame：** RViz Fixed Frame 用 `base_link`，注意光学系轴向。  
5. **4090 + 官方镜像：** 需挂载主机 CUDA≥12 以编译 sm_89（见云端文档）。

## 10. 结论与后续

已完成「标定 → 分割/位姿 → 手眼噪声理解 → 公开数据 GT 量化 → ROS2 离线 TF/bag」感知闭环的工程验证。  

后续可选：真实检测器 bbox、真实手眼采数、再扩展 PCL / MoveIt2 / 学习式策略。

紧凑指标副本：[`results_summary.json`](results_summary.json)。

---

## 附录：口述 / 录屏提纲（3～5 分钟）

1. **目标一句话**（30s）：公开 RGB-D 上的 6D 感知 Pipeline，可量化、可 ROS 回放。  
2. **链路图**（40s）：标定 → FastSAM/GT mask → FoundationPose → `T_base_object` → ROS2。  
3. **关键数字**（60s）：标定重投影 ~0.14 px；YCB ADD-S ~1.6 mm；FastSAM IoU 0.95 且位姿差很小。  
4. **演示**（90s）：WSL launch + RViz（或播放 bag）；指出 TF 与 pose_base。  
5. **边界**（40s）：模拟手眼外参；未做 MoveIt/PCL；云端跑 FP。  

录屏建议窗口：终端 launch 日志 + RViz；保存到本地 `outputs/demo_walkthrough.mp4`（不提交到 Git）。
