# ROS2 offline replay and TF2

Assumes **ROS 2 Jazzy** on WSL2 Ubuntu 24.04 (or native Linux). Replace `$REPO` with your absolute path to this repository.

## Build

```bash
cd $REPO/ros2_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Launch

```bash
ros2 launch robot_pose_pipeline_ros offline_pipeline.launch.py \
  manifest:=$REPO/data/synthetic/manifest.jsonl \
  base_camera_pose:=$REPO/data/synthetic/T_base_camera.txt \
  loop:=true \
  max_frames:=50 \
  rate_hz:=5.0
```

Optional parameters:

- `loop`: loop the manifest (useful for bag recording / demos)
- `max_frames`: `0` = all frames
- `rate_hz`: playback rate
- `mask_mode`: `gt_mask` or `predicted_mask`

Mustard0 example (after poses are attached locally):

```bash
ros2 launch robot_pose_pipeline_ros offline_pipeline.launch.py \
  manifest:=$REPO/data/foundationpose/demo_data/mustard0_ros_manifest.jsonl \
  base_camera_pose:=$REPO/data/foundationpose/demo_data/T_base_camera_demo.txt \
  loop:=true max_frames:=50 rate_hz:=5.0
```

Published topics:

- `/camera/color/image_raw` — RGB `sensor_msgs/Image`
- `/camera/depth/image_raw` — Depth `sensor_msgs/Image`
- `/camera/color/camera_info` — `sensor_msgs/CameraInfo`
- `/perception/mask` — GT or predicted mask
- `/perception/target_cloud` — `sensor_msgs/PointCloud2` from mask + depth
- `/perception/pose_camera` — `T_camera_object`
- `/perception/pose_base` — `T_base_object` via TF2
- `/tf` / `/tf_static` — base → camera → object

Synthetic smoke test (no FoundationPose required):

```powershell
# Windows
python scripts\generate_synthetic_dataset.py
```

```bash
# WSL / Linux
ros2 launch robot_pose_pipeline_ros offline_pipeline.launch.py \
  manifest:=$REPO/data/synthetic/manifest.jsonl \
  base_camera_pose:=$REPO/data/synthetic/T_base_camera.txt
```

## RViz2

```bash
rviz2 -d $REPO/ros2_ws/src/robot_pose_pipeline_ros/rviz/offline_pipeline.rviz
```

Or configure manually: Fixed Frame = `base_link`; add TF, Image (`/camera/color/image_raw`), Pose (`/perception/pose_base`), PointCloud2 (`/perception/target_cloud`). Optical frame axes: x right, y down, z forward.

## Record and replay

```bash
ros2 bag record /camera/color/image_raw /camera/depth/image_raw \
  /camera/color/camera_info /perception/mask /perception/target_cloud \
  /perception/pose_camera /perception/pose_base /tf /tf_static
```

```bash
ros2 bag info <bag_dir>
ros2 bag play <bag_dir> --clock
```

Bag directories under `outputs/` are local-only (gitignored).

## Design boundary

FoundationPose produces pose files on a GPU host (often cloud Docker). This ROS2 package offline-publishes RGB-D, clouds, poses, and TFs for inspection and bagging—without embedding CUDA inference in the node graph.
