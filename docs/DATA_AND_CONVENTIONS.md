# 数据格式与坐标约定

## 唯一位姿约定

`T_parent_child` 把 child 坐标系中的点变换到 parent 坐标系：

```text
p_parent = T_parent_child · p_child
T_base_object = T_base_camera · T_camera_object
```

4×4 齐次矩阵左上角是旋转 `R`（姿态），右上角是平移 `t`（位置）。旋转矩阵必须满足 `RᵀR=I` 且 `det(R)=+1`。

Eye-in-Hand 关系：

```text
T_base_camera(i) = T_base_gripper(i) · T_gripper_camera
T_base_target = T_base_gripper(i) · T_gripper_camera · T_camera_target(i)
```

`T_gripper_camera` 来自手眼标定；`T_camera_object` 来自 FoundationPose；两者不能因为名称相似而颠倒。

## JSONL manifest

每行是一帧，示例见 `config/dataset_manifest.example.jsonl`。字段：

| 字段 | 含义 |
|---|---|
| `frame_id` | 帧唯一编号 |
| `timestamp_sec` | 回放时间戳 |
| `rgb` / `depth` | 彩色图和原始深度图 |
| `camera_matrix` | 3×3 内参 K |
| `depth_scale_to_m` | 深度像素值换算为米的比例 |
| `gt_mask` / `predicted_mask` | 基线和 FastSAM 掩膜 |
| `mesh` | CAD/mesh 路径 |
| `gt_pose` / `predicted_pose` | 4×4 `T_camera_object` 文本文件 |
| `object_id` | 实例或类别编号 |

所有相对路径相对于 manifest 文件本身解析。

## 必做检查

- 深度单位是否由 mm 正确换成 m。
- RGB、Depth、Mask 是否同分辨率、同时间戳并已对齐。
- 内参是否对应当前分辨率；缩放图像时必须同比例缩放 `fx fy cx cy`。
- FoundationPose、数据集标注和 ROS optical frame 的轴方向是否一致。
- mesh 单位是 m 还是 mm；ADD/ADD-S 的输出单位跟 mesh 一致。
- 对称物体优先看 ADD-S，非对称物体看 ADD。
