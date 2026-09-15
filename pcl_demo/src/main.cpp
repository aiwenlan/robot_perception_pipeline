/**
 * Minimal PCL demo for interview / learning.
 *
 * Pipeline (camera frame, meters):
 *   load PCD
 *   -> VoxelGrid
 *   -> PassThrough (Z) + StatisticalOutlierRemoval
 *   -> RANSAC plane removal
 *   -> EuclideanClusterExtraction
 *   -> ICP (largest cluster as target, optional model cloud)
 *
 * Build (WSL):
 *   mkdir -p build && cd build
 *   cmake .. && cmake --build . -j
 *   ./pcl_mini_demo --input ../data/scene.pcd --model ../data/model.pcd --out ../outputs
 */

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <Eigen/Dense>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

struct Args {
  std::string input;
  std::string model;
  std::string out_dir = "outputs";
  float voxel = 0.005f;
  float z_min = 0.1f;
  float z_max = 2.0f;
  int sor_mean_k = 20;
  float sor_stddev = 1.0f;
  float plane_thresh = 0.01f;
  float cluster_tol = 0.02f;
  int cluster_min = 50;
  int cluster_max = 250000;
  float icp_max_corr = 0.02f;
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --input scene.pcd [--model model.pcd] [--out DIR]\n"
      << "  [--voxel 0.005] [--zmin 0.1] [--zmax 2.0]\n"
      << "  [--plane-thresh 0.01] [--cluster-tol 0.02]\n";
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--input") a.input = need("--input");
    else if (k == "--model") a.model = need("--model");
    else if (k == "--out") a.out_dir = need("--out");
    else if (k == "--voxel") a.voxel = std::stof(need("--voxel"));
    else if (k == "--zmin") a.z_min = std::stof(need("--zmin"));
    else if (k == "--zmax") a.z_max = std::stof(need("--zmax"));
    else if (k == "--plane-thresh") a.plane_thresh = std::stof(need("--plane-thresh"));
    else if (k == "--cluster-tol") a.cluster_tol = std::stof(need("--cluster-tol"));
    else if (k == "--help" || k == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "unknown arg: " << k << "\n";
      usage(argv[0]);
      std::exit(2);
    }
  }
  if (a.input.empty()) {
    usage(argv[0]);
    std::exit(2);
  }
  return a;
}

static void save_cloud(const fs::path& path, const CloudT::Ptr& cloud) {
  fs::create_directories(path.parent_path());
  if (pcl::io::savePCDFileBinary(path.string(), *cloud) != 0) {
    throw std::runtime_error("failed to save " + path.string());
  }
  std::cout << "  wrote " << path.string() << "  points=" << cloud->size() << "\n";
}

static CloudT::Ptr voxel_filter(const CloudT::Ptr& in, float leaf) {
  CloudT::Ptr out(new CloudT);
  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(in);
  vg.setLeafSize(leaf, leaf, leaf);
  vg.filter(*out);
  return out;
}

static CloudT::Ptr passthrough_z(const CloudT::Ptr& in, float zmin, float zmax) {
  CloudT::Ptr out(new CloudT);
  pcl::PassThrough<PointT> pass;
  pass.setInputCloud(in);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(zmin, zmax);
  pass.filter(*out);
  return out;
}

static CloudT::Ptr statistical_outlier(const CloudT::Ptr& in, int mean_k, float stddev) {
  CloudT::Ptr out(new CloudT);
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(in);
  sor.setMeanK(mean_k);
  sor.setStddevMulThresh(stddev);
  sor.filter(*out);
  return out;
}

static CloudT::Ptr remove_plane(const CloudT::Ptr& in, float dist_thresh, Eigen::Vector4f* plane_out) {
  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(dist_thresh);
  seg.setInputCloud(in);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
  seg.segment(*inliers, *coeff);

  if (inliers->indices.empty()) {
    std::cout << "  [warn] no plane found; skipping plane removal\n";
    if (plane_out) plane_out->setZero();
    return in;
  }
  if (plane_out && coeff->values.size() >= 4) {
    (*plane_out) << coeff->values[0], coeff->values[1], coeff->values[2], coeff->values[3];
  }
  std::cout << "  plane inliers=" << inliers->indices.size()
            << "  model=[" << coeff->values[0] << " " << coeff->values[1] << " "
            << coeff->values[2] << " " << coeff->values[3] << "]\n";

  // Only remove plane if it dominates; otherwise keep original (object-only scenes).
  const float ratio = static_cast<float>(inliers->indices.size()) / static_cast<float>(in->size());
  if (ratio < 0.25f) {
    std::cout << "  plane ratio=" << ratio << " < 0.25; keep cloud unchanged\n";
    return in;
  }

  CloudT::Ptr outliers(new CloudT);
  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(in);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*outliers);
  return outliers;
}

static std::vector<CloudT::Ptr> euclidean_clusters(
    const CloudT::Ptr& in, float tol, int min_size, int max_size) {
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(in);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(tol);
  ec.setMinClusterSize(min_size);
  ec.setMaxClusterSize(max_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(in);
  ec.extract(cluster_indices);

  std::vector<CloudT::Ptr> clusters;
  clusters.reserve(cluster_indices.size());
  for (const auto& indices : cluster_indices) {
    CloudT::Ptr cloud(new CloudT);
    cloud->reserve(indices.indices.size());
    for (int idx : indices.indices) {
      cloud->push_back((*in)[idx]);
    }
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    clusters.push_back(cloud);
  }
  return clusters;
}

static void run_icp(const CloudT::Ptr& source, const CloudT::Ptr& target, float max_corr,
                    Eigen::Matrix4f* T_out, double* fitness_out, double* score_out) {
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setInputSource(source);
  icp.setInputTarget(target);
  icp.setMaxCorrespondenceDistance(max_corr);
  icp.setMaximumIterations(50);
  icp.setTransformationEpsilon(1e-8);
  icp.setEuclideanFitnessEpsilon(1e-8);

  CloudT aligned;
  icp.align(aligned);
  *T_out = icp.getFinalTransformation();
  *fitness_out = icp.hasConverged() ? 1.0 : 0.0;
  *score_out = icp.getFitnessScore();
  std::cout << "  ICP converged=" << (icp.hasConverged() ? "true" : "false")
            << "  fitness_score(MSE)=" << *score_out << "\n";
  std::cout << "  T_target_source =\n" << *T_out << "\n";
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    fs::path out_dir = args.out_dir;
    fs::create_directories(out_dir);

    CloudT::Ptr cloud(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(args.input, *cloud) < 0 || cloud->empty()) {
      std::cerr << "failed to load input: " << args.input << "\n";
      return 1;
    }
    std::cout << "[0] load " << args.input << " points=" << cloud->size() << "\n";
    save_cloud(out_dir / "00_input.pcd", cloud);

    auto vox = voxel_filter(cloud, args.voxel);
    std::cout << "[1] VoxelGrid leaf=" << args.voxel << " points=" << vox->size() << "\n";
    save_cloud(out_dir / "01_voxel.pcd", vox);

    auto zcut = passthrough_z(vox, args.z_min, args.z_max);
    std::cout << "[2] PassThrough z=[" << args.z_min << "," << args.z_max << "] points=" << zcut->size()
              << "\n";
    save_cloud(out_dir / "02_passthrough.pcd", zcut);

    auto clean = statistical_outlier(zcut, args.sor_mean_k, args.sor_stddev);
    std::cout << "[3] SOR meanK=" << args.sor_mean_k << " std=" << args.sor_stddev
              << " points=" << clean->size() << "\n";
    save_cloud(out_dir / "03_sor.pcd", clean);

    Eigen::Vector4f plane_coeff = Eigen::Vector4f::Zero();
    auto rest = remove_plane(clean, args.plane_thresh, &plane_coeff);
    std::cout << "[4] after plane points=" << rest->size() << "\n";
    save_cloud(out_dir / "04_no_plane.pcd", rest);

    if (rest->empty()) {
      std::cerr << "empty cloud after plane removal\n";
      return 1;
    }

    auto clusters = euclidean_clusters(rest, args.cluster_tol, args.cluster_min, args.cluster_max);
    std::cout << "[5] Euclidean clusters=" << clusters.size() << "\n";
    for (size_t i = 0; i < clusters.size(); ++i) {
      PointT mn{}, mx{};
      pcl::getMinMax3D(*clusters[i], mn, mx);
      std::cout << "  cluster[" << i << "] n=" << clusters[i]->size()
                << " aabb=(" << mn.x << "," << mn.y << "," << mn.z << ")->("
                << mx.x << "," << mx.y << "," << mx.z << ")\n";
      save_cloud(out_dir / ("05_cluster_" + std::to_string(i) + ".pcd"), clusters[i]);
    }
    if (clusters.empty()) {
      std::cerr << "no clusters found\n";
      return 1;
    }

    // Largest cluster as object observation (target for ICP).
    size_t best_i = 0;
    for (size_t i = 1; i < clusters.size(); ++i) {
      if (clusters[i]->size() > clusters[best_i]->size()) best_i = i;
    }
    CloudT::Ptr object = clusters[best_i];
    save_cloud(out_dir / "06_object_cluster.pcd", object);
    std::cout << "[6] object cluster index=" << best_i << " points=" << object->size() << "\n";

    CloudT::Ptr model(new CloudT);
    if (!args.model.empty()) {
      if (pcl::io::loadPCDFile<PointT>(args.model, *model) < 0 || model->empty()) {
        std::cerr << "failed to load model: " << args.model << "\n";
        return 1;
      }
    } else {
      // No external model: copy object and apply a small known perturbation as "source".
      *model = *object;
      Eigen::Matrix4f noise = Eigen::Matrix4f::Identity();
      noise(0, 3) = 0.015f;
      noise(1, 3) = -0.010f;
      noise(2, 3) = 0.005f;
      pcl::transformPointCloud(*model, *model, noise);
      std::cout << "[7] synthesized model = object + translation noise\n";
    }
    save_cloud(out_dir / "07_model_source.pcd", model);

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    double fitness = 0.0;
    double score = 0.0;
    std::cout << "[8] ICP  (p_target ≈ T_target_source * p_source)\n";
    run_icp(model, object, args.icp_max_corr, &T, &fitness, &score);

    CloudT::Ptr aligned(new CloudT);
    pcl::transformPointCloud(*model, *aligned, T);
    save_cloud(out_dir / "08_model_after_icp.pcd", aligned);

    std::cout << "DONE. outputs in " << out_dir << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
