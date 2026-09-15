# Open3D ↔ PCL Equivalents

本仓库算法路径优先 **Open3D（Python）**。下表便于面试时映射到 PCL 常用接口（C++）。  
未提供完整 PCL 工程；若本机无 PCL，以本表 + Open3D demo 为准。

| 能力 | Open3D（本仓） | PCL 常用类 / 接口 |
|---|---|---|
| 点云容器 | `open3d.geometry.PointCloud` | `pcl::PointCloud<pcl::PointXYZ>` / `PointXYZRGB` |
| Voxel 下采样 | `voxel_down_sample` / `pointcloud.voxel_downsample` | `pcl::VoxelGrid` |
| Statistical outlier | `remove_statistical_outlier` | `pcl::StatisticalOutlierRemoval` |
| Radius outlier | `remove_radius_outlier` | `pcl::RadiusOutlierRemoval` |
| Crop / passthrough | `AxisAlignedBoundingBox` + `crop` | `pcl::CropBox` / `pcl::PassThrough` |
| 法向 | `estimate_normals` | `pcl::NormalEstimation` / `NormalEstimationOMP` |
| RANSAC 平面 | `segment_plane` | `pcl::SACSegmentation` + `SACMODEL_PLANE` |
| 欧氏聚类 | （可用 DBSCAN 近似） | `pcl::EuclideanClusterExtraction` |
| DBSCAN | `cluster_dbscan` | PCL 无官方同名；常用欧氏聚类或第三方 |
| ICP point-to-point | `TransformationEstimationPointToPoint` | `pcl::IterativeClosestPoint` |
| ICP point-to-plane | `TransformationEstimationPointToPlane` | `pcl::IterativeClosestPointWithNormals` |
| GICP | Open3D 有相关注册（本仓未默认启用） | `pcl::GeneralizedIterativeClosestPoint` |
| Mesh 采样 | `sample_points_uniformly` | `pcl::UniformSampling` / 从 `PolygonMesh` 采样 |
| TSDF / 积分 | `ScalableTSDFVolume` | `pcl::KinFu` / `gpu::KinFu`（生态不同，勿 1:1） |
| IO PLY | `open3d.io.write_point_cloud` | `pcl::io::savePLYFile` |

## 最小 C++/PCL demo（可选）

若已安装 PCL，可自建单文件：读 PLY → `VoxelGrid` → 保存。本仓不强制编入 CI，避免绑架 Windows/WSL 环境。

```cpp
// sketch only — not built in this repo by default
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
int main() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPLYFile("object.ply", *cloud);
  pcl::VoxelGrid<pcl::PointXYZ> vg; vg.setInputCloud(cloud); vg.setLeafSize(0.005f,0.005f,0.005f);
  pcl::PointCloud<pcl::PointXYZ> out; vg.filter(out);
  pcl::io::savePLYFileBinary("object_voxel.ply", out);
}
```

## 面试怎么说
“项目验证用 Open3D 走通滤波/平面/ICP/TSDF；概念与 PCL 一一对应，量产机器人栈再迁 C++/PCL 即可。”
