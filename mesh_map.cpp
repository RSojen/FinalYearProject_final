#include "aru/core/mapping/mesh_mapping/mesh_map.h"

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>
#include <utility>

using namespace aru::core::utilities;
using namespace aru::core::utilities::image;
using namespace voxblox;
namespace aru {
namespace core {
namespace mapping {
namespace mesh_map {
//------------------------------------------------------------------------------
MeshMap::MeshMap(std::string mesh_map_settings_file)
    : mesh_map_settings_file_(std::move(mesh_map_settings_file)) {

  Eigen::Affine3f identity_affine;
  identity_affine.matrix() = Eigen::MatrixXf::Identity(4, 4);
  curr_pose_ = transform::Transform(0, 0, identity_affine);
  cv::FileStorage fs;
  fs.open(mesh_map_settings_file_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Could not open mesh map settings file: ";
  }
  // Regularisor choice
  fs["Regulariser"]["TV"] >> reg_params_.use_tv;
  fs["Regulariser"]["TGV"] >> reg_params_.use_tgv;
  fs["Regulariser"]["LOG_TV"] >> reg_params_.use_log_tv;
  fs["Regulariser"]["LOG_TGV"] >> reg_params_.use_log_tgv;

  // Regularisation parameters
  reg_params_.sigma = fs["Regulariser"]["sigma"];
  reg_params_.tau = fs["Regulariser"]["tau"];
  reg_params_.lambda = fs["Regulariser"]["lambda"];
  reg_params_.theta = fs["Regulariser"]["theta"];
  reg_params_.alpha_1 = fs["Regulariser"]["alpha_1"];
  reg_params_.alpha_2 = fs["Regulariser"]["alpha_2"];
  reg_params_.beta = fs["Regulariser"]["beta"];
  reg_params_.iterations = fs["Regulariser"]["iterations"];
  reg_params_.outer_iterations = fs["Regulariser"]["outer_iterations"];

  // Viewer params
  viewer_params_.max_depth = fs["Viewer"]["max_depth"];
  viewer_params_.colour_scale = fs["Viewer"]["colour_scale"];

  // Camera params
  LOG(INFO) << "Camera params found";
  camera_params_.baseline = fs["Camera"]["baseline"];
  camera_params_.image_height = fs["Camera"]["height"];
  camera_params_.image_width = fs["Camera"]["width"];
  cv::Mat camera_mat;
  fs["Camera"]["CameraMatrix"] >> camera_mat;
  cv::cv2eigen(camera_mat, camera_params_.K);

  // Matcher params
  matcher_params_.focal_length = fs["FeatureMatcher"]["focal_length"];
  matcher_params_.stereo_baseline = fs["FeatureMatcher"]["stereo_baseline"];
  matcher_params_.match_threshold_low = fs["FeatureMatcher"
                                           ""]["match_threshold_low"];
  matcher_params_.match_threshold_high = fs["FeatureMatcher"
                                            ""]["match_threshold_high"];

  // Extractor params
  extractor_params_.patch_size = fs["FeatureExtractor"]["patch_size"];
  extractor_params_.half_patch_size = fs["FeatureExtractor"]["half_patch_size"];
  extractor_params_.num_levels = fs["FeatureExtractor"]["num_levels"];
  extractor_params_.scale_factor = fs["FeatureExtractor"]["scale_factor"];
  extractor_params_.edge_threshold = fs["FeatureExtractor"]["edge_threshold"];
  extractor_params_.num_features = fs["FeatureExtractor"]["num_features"];
  extractor_params_.initial_fast_threshold = fs["FeatureExtractor"
                                                ""]["initial_fast_threshold"];
  extractor_params_.minimum_fast_threshold = fs["FeatureExtractor"
                                                ""]["minimum_fast_threshold"];

  // Solver Parameters
  solver_params_.ransac_prob = fs["FeatureSolver"]["ransac_prob"];
  solver_params_.ransac_max_iterations =
      fs["FeatureSolver"]["ransac_max_iterations"];
  solver_params_.threshold = fs["FeatureSolver"]["inlier_threshold"];

  // Camera params
  cv::Mat vo_camera_mat;
  fs["FeatureSolver"]["CameraMatrix"] >> vo_camera_mat;
  cv::cv2eigen(vo_camera_mat, solver_params_.camera_matrix);


  mesh_ = boost::make_shared<mesh::Mesh>(reg_params_, viewer_params_,
                                         camera_params_, matcher_params_,
                                         extractor_params_);
  LOG(INFO) << "Mesh Pointer Created";

  viewer_ = boost::make_shared<utilities::viewer::Viewer>(
      camera_params_.image_height, camera_params_.image_width,
      camera_params_.K);

  LOG(INFO) << "viewer pointer created";

  vo_ = boost::make_shared<aru::core::vo::VO>(extractor_params_,
                                              matcher_params_, solver_params_);
  LOG(INFO) << "visual odometrey pointer created";

  // Initialise VISO
  viso_extractor_ = boost::make_shared<utilities::image::VisoFeatureTracker>(
      matcher_params_, extractor_params_);

  orb_matcher_ = boost::make_shared<utilities::image::OrbFeatureMatcher>(
      matcher_params_, extractor_params_,
      "/home/ritvik/Data/husky_data/vocabulary/ORBvoc.txt");

  voxblox::TsdfMap::Config config_;
  config_.tsdf_voxel_size = 0.1;
  config_.tsdf_voxels_per_side = 32;

  voxblox::TsdfIntegratorBase::Config base_config{};
  base_config.max_ray_length_m = 5;

  tsdf_map_ = boost::make_shared<voxblox::TsdfMap>(config_);
  tsdf_integrator_ = boost::make_shared<voxblox::FastTsdfIntegrator>(
      base_config, tsdf_map_->getTsdfLayerPtr());

  mesh_layer_ = boost::make_shared<voxblox::MeshLayer>(tsdf_map_->block_size());
  LOG(INFO) << "Block size is " << tsdf_map_->block_size();
  mesh_integrator_ =
      boost::make_shared<voxblox::MeshIntegrator<voxblox::TsdfVoxel>>(
          MeshIntegratorConfig(),tsdf_map_->getTsdfLayerPtr(), mesh_layer_.get());

  icp_ = boost::make_shared<voxblox::ICP>(voxblox::ICP::Config());
  use_laser_ = false;
  transform_map_ = boost::make_shared<utilities::transform::TransformMap>();

  // Initialise the mesh
  mesh_->InitXYZ();
}
//------------------------------------------------------------------------------
void MeshMap::InitialiseMap(utilities::image::StereoImage image_init) {
  // Get the features in the current frame
  use_laser_ = false;
  frame_no = 0;
  cv::Mat image_1_left_grey, image_1_right_grey;

  cv::cvtColor(image_init.first.GetImage(), image_1_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_init.second.GetImage(), image_1_right_grey,
               cv::COLOR_BGR2GRAY);

  //    frame_features_ = curvature_matcher_->ComputeStereoMatches(
  //        image_1_left_grey, image_1_right_grey);

  LOG(INFO) << "Computing the stereo matches";
  //  viso_extractor_->FeaturesUntracked(image_1_left_grey, image_1_right_grey);
  //  frame_features_ = viso_extractor_->GetCurrentFeatures();

  frame_features_ =
      orb_matcher_->ComputeStereoMatches(image_1_left_grey, image_1_right_grey);
  // Save the current frame
  curr_frame_ =
      image::StereoImage{image::Image(frame_no, image_init.first.GetImage()),
                         image::Image(frame_no, image_init.second.GetImage())};

  // Initialise the position
  current_position_.matrix() = Eigen::MatrixXf::Identity(4, 4);

  // Calculate the mesh at this frame
  // UpdateFrameMesh();
}
//------------------------------------------------------------------------------
void MeshMap::ReadTransform(utilities::transform::TransformSPtr transform) {
  transform_map_->AddTransform(std::move(transform));
}
//------------------------------------------------------------------------------
void MeshMap::UpdateFrameMesh() {

  cv::Mat image_1_left_grey, image_1_right_grey;

  cv::cvtColor(curr_frame_.first.GetImage(), image_1_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(curr_frame_.second.GetImage(), image_1_right_grey,
               cv::COLOR_BGR2GRAY);
  bool triangulate = true;
  mesh_->EstimateMesh(frame_features_, triangulate);
  // Draw the mesh
  cv::Mat disparity = mesh_->GetInterpolatedDepth();
  std::vector<cv::KeyPoint> keypoints = mesh_->GetVerticeKeypoints();
  std::vector<double> depths = mesh_->GetVerticeDepths();
  std::vector<Eigen::Vector3i> triangles = mesh_->GetMeshTriangles();

  cv::Mat image_clone = curr_frame_.first.GetImage().clone();
  viewer_->ViewMeshWireFrame(image_clone, keypoints, depths, triangles,
                             viewer_params_.max_depth);
  viewer_->ViewInterpolatedMesh(curr_frame_.first.GetImage(), disparity,
                                viewer_params_.max_depth);
  cv::waitKey(1);

  if (frame_no % 1 == 0) {
    float max_depth = 10;
    auto estimation_start = std::chrono::high_resolution_clock::now();
    auto pt_cloud_colour = mesh_->GetInterpolatedColorPointCloud(
        curr_frame_.first.GetImage(), max_depth);
    voxblox::Pointcloud pointcloud_float = pt_cloud_colour.first;
    voxblox::Colors colors = pt_cloud_colour.second;

    auto estimation_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = estimation_end - estimation_start;
    VLOG(2) << "Interpolation takes " << elapsed.count() << " seconds";

    //    voxblox::Pointcloud pointcloud_float;
    //    voxblox::Colors colors;
    //    //    Eigen::Matrix3f rotation;
    //    //    rotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;
    //    for (const auto &vertice : curr_pointcloud_) {
    //      Eigen::Vector3f point_f = vertice.cast<float>();
    //      pointcloud_float.push_back(point_f);
    //    }
    //    for (const auto &point_color : color_pointcloud_) {
    //      voxblox::Color color(point_color(0), point_color(1),
    //                                        point_color(2));
    //      colors.push_back(color);
    //    }
    Eigen::Matrix3f rot = current_position_.linear();
    Eigen::Vector3f pos = current_position_.translation();
    Transformation icp_initial(Rotation(rot), current_position_.translation());
    //    Transformation icp_refine;
    //    icp_->runICP(tsdf_map_->getTsdfLayer(), pointcloud_float, icp_initial,
    //                 &icp_refine);

    auto integration_start = std::chrono::high_resolution_clock::now();
    tsdf_integrator_->integratePointCloud(icp_initial, pointcloud_float,
                                          colors);
    auto integration_end = std::chrono::high_resolution_clock::now();
    elapsed = integration_end - integration_start;
    VLOG(2) << "Interpolation takes " << elapsed.count() << " seconds";
    // current_position_.matrix() = icp_refine.getTransformationMatrix();
  }
}

//------------------------------------------------------------------------------
void MeshMap::InsertDepthImage(cv::Mat depth_image, cv::Mat rgb_image,
                               Eigen::Affine3f position) {
  //LOG(INFO) << "Adding depth image";
  auto pt_cloud_colour =
      mesh_->GetInterpolatedColorPointCloud(rgb_image, depth_image);
  //viewer_->ViewDepthPointCloud(depth_image, 50);
  //cv::waitKey(0);
  voxblox::Pointcloud pointcloud_float = pt_cloud_colour.first;
  voxblox::Colors colors = pt_cloud_colour.second;

  Eigen::Matrix3f rot = position.rotation();
  Eigen::Vector3f pos = position.translation();
  Transformation icp_initial(Rotation(rot), pos);
  //LOG(INFO)<<"Integrating depth image";
  tsdf_integrator_->integratePointCloud(icp_initial, pointcloud_float, colors);
  //LOG(INFO)<<"Added depth image";
}
//------------------------------------------------------------------------------
void MeshMap::InsertDepthImage(utilities::image::Image depth_image,
                                                              utilities::image::Image rgb_image) {
  //LOG(INFO) << "Adding depth image";
  auto reading_start = std::chrono::high_resolution_clock::now();
  cv::Mat disparity = depth_image.GetImage();
  cv::Mat image_clone = rgb_image.GetImage().clone();
  cv::Mat depth = mesh_->DisparityToDepth(disparity);
  auto reading_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = reading_end - reading_start;
  //LOG(INFO) << "Reading takes " << elapsed.count() << " seconds";
  //viewer_->ViewDisparity(disparity, 50);
  //viewer_->ViewInterpolatedMesh(image_clone, depth, viewer_params_.max_depth);
  cv::waitKey(1);

  float max_depth = 10;
  auto estimation_start = std::chrono::high_resolution_clock::now();

  auto pt_cloud_colour =
      mesh_->GetInterpolatedColorPointCloud(rgb_image.GetImage(), depth);
  voxblox::Pointcloud pointcloud_float = pt_cloud_colour.first;
  voxblox::Colors colors = pt_cloud_colour.second;

  auto estimation_end = std::chrono::high_resolution_clock::now();
  elapsed = estimation_end - estimation_start;
  //LOG(INFO) << "Interpolation takes " << elapsed.count() << " seconds";

  //  voxblox::Pointcloud pointcloud_float;
  //  voxblox::Colors colors;
  //  //    Eigen::Matrix3f rotation;
  //  //    rotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;
  //  for (const auto &vertice : curr_pointcloud_) {
  //    Eigen::Vector3f point_f = vertice.cast<float>();
  //    pointcloud_float.push_back(point_f);
  //  }
  //  for (const auto &point_color : color_pointcloud_) {
  //    voxblox::Color color(point_color(0), point_color(1),
  //                                      point_color(2));
  //    colors.push_back(color);
  //  }
  //LOG(INFO) << transform_map_;
  utilities::transform::TransformSPtr curr_position =
      transform_map_->Interpolate(depth_image.GetTimeStamp());
  //LOG(INFO) << curr_position;
  if (curr_position) {
    Eigen::Matrix3f rot = curr_position->GetRotation();
    Eigen::Vector3f pos = curr_position->GetTranslation();

    Transformation icp_initial(Rotation(rot), pos);

    auto integration_start = std::chrono::high_resolution_clock::now();
    tsdf_integrator_->integratePointCloud(icp_initial, pointcloud_float,
                                          colors);
    auto integration_end = std::chrono::high_resolution_clock::now();
    elapsed = integration_end - integration_start;
    //LOG(INFO) << "Integration takes " << elapsed.count() << " seconds";
  }
  
  // current_position_.matrix() = icp_refine.getTransformationMatrix();
}
//------------------------------------------------------------------------------
void MeshMap::SaveCurrentTsdf(std::string output_ply) {

  constexpr bool kOnlyMeshUpdatedBlocks = false;
  constexpr bool kClearUpdatedFlag = false;
  mesh_integrator_->generateMesh(kOnlyMeshUpdatedBlocks, kClearUpdatedFlag);

  BlockIndexList mesh_indices;
  mesh_layer_->getAllAllocatedMeshes(&mesh_indices);
  std::vector<Eigen::Vector3d> voxel_points_;
  std::vector<Eigen::Vector3d> color_points_;
  // Write to ply
  std::ofstream stream(output_ply);

  Mesh combined_mesh(mesh_layer_->block_size(), Point::Zero());

  mesh_layer_->getConnectedMesh(&combined_mesh);

  size_t num_points = combined_mesh.vertices.size();
  stream << "ply" << std::endl;
  stream << "format ascii 1.0" << std::endl;
  stream << "element vertex " << num_points << std::endl;
  stream << "property float x" << std::endl;
  stream << "property float y" << std::endl;
  stream << "property float z" << std::endl;
  if (combined_mesh.hasNormals()) {
    stream << "property float normal_x" << std::endl;
    stream << "property float normal_y" << std::endl;
    stream << "property float normal_z" << std::endl;
  }
  if (combined_mesh.hasColors()) {
    stream << "property uchar red" << std::endl;
    stream << "property uchar green" << std::endl;
    stream << "property uchar blue" << std::endl;
    stream << "property uchar alpha" << std::endl;
  }
  if (combined_mesh.hasTriangles()) {
    stream << "element face " << combined_mesh.indices.size() / 3 << std::endl;
    stream << "property list uchar int vertex_indices"
           << std::endl; // pcl-1.7(ros::kinetic) breaks ply convention by not
    // using "vertex_index"
  }
  stream << "end_header" << std::endl;
  size_t vert_idx = 0;
  for (const Point &vert : combined_mesh.vertices) {
    stream << vert(0) << " " << vert(1) << " " << vert(2);
    Eigen::Vector3d point;
    point.x() = vert(0);
    point.y() = vert(1);
    point.z() = vert(2);
    voxel_points_.push_back(point);

    if (combined_mesh.hasNormals()) {
      const Point &normal = combined_mesh.normals[vert_idx];
      stream << " " << normal.x() << " " << normal.y() << " " << normal.z();
    }
    if (combined_mesh.hasColors()) {
      const Color &color = combined_mesh.colors[vert_idx];
      Eigen::Vector3d color_point(color.r, color.g, color.b);
      color_points_.push_back(color_point);
      int r = static_cast<int>(color.r);
      int g = static_cast<int>(color.g);
      int b = static_cast<int>(color.b);
      int a = static_cast<int>(color.a);
      // Uint8 prints as character otherwise. :(
      stream << " " << r << " " << g << " " << b << " " << a;
    }

    stream << std::endl;
    vert_idx++;
  }
  if (combined_mesh.hasTriangles()) {
    for (size_t i = 0; i < combined_mesh.indices.size(); i += 3) {
      stream << "3 ";

      for (int j = 0; j < 3; j++) {
        stream << combined_mesh.indices.at(i + j) << " ";
      }

      stream << std::endl;
    }
  }
}

//------------------------------------------------------------------------------
void MeshMap::PoseAndMeshSaveCurrentTsdf(std::string output_ply_unopt, std::string output_ply_opt, DeformationGraph& graph, pcl::PointCloud<pcl::PointXYZ>::Ptr simplified) {

  constexpr bool kOnlyMeshUpdatedBlocks = false;
  constexpr bool kClearUpdatedFlag = false;
  mesh_integrator_->generateMesh(kOnlyMeshUpdatedBlocks, kClearUpdatedFlag);

  BlockIndexList mesh_indices;
  mesh_layer_->getAllAllocatedMeshes(&mesh_indices);
  std::vector<Eigen::Vector3d> voxel_points_;
  std::vector<Eigen::Vector3d> color_points_;
  // Write to ply files
  std::ofstream unopt_stream(output_ply_unopt);
  std::ofstream opt_stream(output_ply_opt);

  Mesh combined_mesh(mesh_layer_->block_size(), Point::Zero());

  mesh_layer_->getConnectedMesh(&combined_mesh);

  size_t num_points = combined_mesh.vertices.size();
  unopt_stream << "ply" << std::endl;
  unopt_stream << "format ascii 1.0" << std::endl;
  unopt_stream << "element vertex " << num_points << std::endl;
  unopt_stream << "property float x" << std::endl;
  unopt_stream << "property float y" << std::endl;
  unopt_stream << "property float z" << std::endl;

  //repeat for optimised file
  opt_stream << "ply" << std::endl;
  opt_stream << "format ascii 1.0" << std::endl;
  opt_stream << "element vertex " << num_points << std::endl;
  opt_stream << "property float x" << std::endl;
  opt_stream << "property float y" << std::endl;
  opt_stream << "property float z" << std::endl;
  if (combined_mesh.hasNormals()) {
    unopt_stream << "property float normal_x" << std::endl;
    unopt_stream << "property float normal_y" << std::endl;
    unopt_stream << "property float normal_z" << std::endl;

    //repeat for optimised file
    opt_stream << "property float normal_x" << std::endl;
    opt_stream << "property float normal_y" << std::endl;
    opt_stream << "property float normal_z" << std::endl;
  }
  if (combined_mesh.hasColors()) {
    unopt_stream << "property uchar red" << std::endl;
    unopt_stream << "property uchar green" << std::endl;
    unopt_stream << "property uchar blue" << std::endl;
    unopt_stream << "property uchar alpha" << std::endl;

    //repeat for optimised file
    opt_stream << "property uchar red" << std::endl;
    opt_stream << "property uchar green" << std::endl;
    opt_stream << "property uchar blue" << std::endl;
    opt_stream << "property uchar alpha" << std::endl;
  }
  if (combined_mesh.hasTriangles()) {
    unopt_stream << "element face " << combined_mesh.indices.size() / 3 << std::endl;
    unopt_stream << "property list uchar int vertex_indices"
           << std::endl; // pcl-1.7(ros::kinetic) breaks ply convention by not
    // using "vertex_index"

    //repeat for optimised file
    opt_stream << "element face " << combined_mesh.indices.size() / 3 << std::endl;
    opt_stream << "property list uchar int vertex_indices" << std::endl;
  }
  unopt_stream << "end_header" << std::endl;
  opt_stream << "end_header" << std::endl;
  size_t vert_idx = 0;
  LOG(INFO) << "number of vertices: " << combined_mesh.vertices.size();
  int counter=0;
  for (const Point &vert : combined_mesh.vertices) {

    Eigen::Vector3d point_deformed;
    Eigen::Vector3d point;

    pcl::PointXYZ vertex(vert(0),vert(1),vert(2));
    point_deformed = graph.DeformVertex(vertex,&counter);
    //LOG(INFO) << "deformed point: ["<<point_deformed.x()<<","<<point_deformed.y()<<","<<point_deformed.z()<<"]";

    point.x() = vert(0);
    point.y() = vert(1);
    point.z() = vert(2);


    unopt_stream << point.x() << " " << point.y() << " " << point.z();
    opt_stream << point_deformed.x() << " " << point_deformed.y() << " " << point_deformed.z();

    voxel_points_.push_back(point);

    if (combined_mesh.hasNormals()) {
      const Point &normal = combined_mesh.normals[vert_idx];
      unopt_stream << " " << normal.x() << " " << normal.y() << " " << normal.z();

      opt_stream << " " << normal.x() << " " << normal.y() << " " << normal.z();
    }
    if (combined_mesh.hasColors()) {
      const Color &color = combined_mesh.colors[vert_idx];
      Eigen::Vector3d color_point(color.r, color.g, color.b);
      color_points_.push_back(color_point);
      int r = static_cast<int>(color.r);
      int g = static_cast<int>(color.g);
      int b = static_cast<int>(color.b);
      int a = static_cast<int>(color.a);
      // Uint8 prints as character otherwise. :(
      unopt_stream << " " << r << " " << g << " " << b << " " << a;
      opt_stream << " " << r << " " << g << " " << b << " " << a;
    }

    unopt_stream << std::endl;
    opt_stream << std::endl;
    vert_idx++;
  }
  LOG(INFO) << "overlapped vertices: " << counter;
  if (combined_mesh.hasTriangles()) {
    for (size_t i = 0; i < combined_mesh.indices.size(); i += 3) {
      unopt_stream << "3 ";
      opt_stream << "3 ";

      for (int j = 0; j < 3; j++) {
        unopt_stream << combined_mesh.indices.at(i + j) << " ";
        opt_stream << combined_mesh.indices.at(i + j) << " ";
      }

      unopt_stream << std::endl;
      opt_stream << std::endl;
    }
  }
}

//------------------------------------------------------------------------------
/*function to incrementally build the deformation graph */
void MeshMap::BuildDfgFromTsdf(DeformationGraph& graph, int poseIndex, double voxelDim, bool loop_closed){

  //generate the mesh with only the updated blocks and clear the updated flags in the tsdf layer
  mesh_integrator_->generateMesh(true, true);

  //variable to store all updated mesh blocks
  BlockIndexList updated_mesh_blocks;
  mesh_layer_->getAllUpdatedMeshes(&updated_mesh_blocks);

  Mesh updated_mesh_full(mesh_layer_->block_size(), Point::Zero());
  Mesh updated_mesh_simplified(mesh_layer_->block_size(), Point::Zero());

  LOG(INFO) << updated_mesh_blocks.size() << " updated mesh blocks";

  //create an association between block index and pose
  /*for (BlockIndex& index : updated_tsdf_indices)
  {
      if(block_map.count(index))
      {
          block_map.at(index).push_back(poseIndex);
      }
      else
      {
          std::vector<int> poses;
          poses.push_back(poseIndex);
          block_map.insert(std::pair<BlockIndex,std::vector<int> >(index,poses));
      }
  }*/
  mesh_layer_->getConnectedMesh(&updated_mesh_full);
  mesh_layer_->getConnectedMesh(&updated_mesh_simplified,voxelDim);

  //std::vector<pcl::PointXYZ> new_vertices;
  std::vector<pcl::PointXYZ> new_nodes;

  //fuse simplified mesh observations
  for (const Point& vert : updated_mesh_simplified.vertices)
  {
      pcl::PointXYZ node(vert(0),vert(1),vert(2));
      //fuse node into current pointcloud
      graph.FuseMeshNode(node,poseIndex,new_nodes,voxelDim);
  }
  LOG(INFO) << "fused mesh nodes";

  //add new nodes
  for(pcl::PointXYZ point : new_nodes)
  {
      graph.addToSimpCloud(point);
  }

  //clear the updated flags for the mesh blocks
  for (BlockIndex& index : updated_mesh_blocks)
  {
      mesh_layer_->getMeshByIndex(index).updated = false;
  }

  /*if(loop_closed)
  {
      mesh_integrator_->generateMesh(false, false);

      Mesh updated_mesh(mesh_layer_->block_size(), Point::Zero());

      mesh_layer_->getConnectedMesh(&updated_mesh);

      int pointIndx=0;

      //add to cumulative cloud
      for (const Point &vert : updated_mesh.vertices) {
          pcl::PointXYZ point(vert(0),vert(1),vert(2));
          //add to cumulative pointcloud
          graph.addToCloud(point);
          //get associated poses
          BlockIndex indx = mesh_layer_->computeBlockIndexFromCoordinates(vert);
          std::vector<int> connected_poses;
          connected_poses=block_map.at(indx);
          //add to pose-mesh correspondences
          for (int pose : connected_poses)
          {
              graph.AddVertexPoseCorr(pointIndx, pose);
          }
          pointIndx++;
      }
*/
  

}
//------------------------------------------------------------------------------

void MeshMap::DrawCurrentTsdf() {

    BlockIndexList tsdf_indices;
    tsdf_map_->getTsdfLayer().getAllUpdatedBlocks(Update::kMesh,&tsdf_indices);

    BlockIndexList alloc_tsdf_indices;
    tsdf_map_->getTsdfLayer().getAllAllocatedBlocks(&alloc_tsdf_indices);


    LOG(INFO) << tsdf_indices.size() << " updated tsdf blocks";
    LOG(INFO) << alloc_tsdf_indices.size() << " allocated tsdf blocks";
    
  constexpr bool kOnlyMeshUpdatedBlocks = false;
  constexpr bool kClearUpdatedFlag = false;

  mesh_integrator_->generateMesh(kOnlyMeshUpdatedBlocks, kClearUpdatedFlag);

  std::vector<Eigen::Vector3d> voxel_points_;
  std::vector<Eigen::Vector3d> color_points_;

  Mesh updated_mesh(mesh_layer_->block_size(), Point::Zero());

  mesh_layer_->getConnectedMesh(&updated_mesh);

  size_t num_points = updated_mesh.vertices.size();
  size_t vert_idx = 0;
  for (const Point &vert : updated_mesh.vertices) {
    Eigen::Vector3d point;
    point.x() = vert(0);
    point.y() = vert(1);
    point.z() = vert(2);
    voxel_points_.push_back(point);

    if (updated_mesh.hasColors()) {
      const Color &color = updated_mesh.colors[vert_idx];
      Eigen::Vector3d color_point(color.r, color.g, color.b);
      color_points_.push_back(color_point);
      int r = static_cast<int>(color.r);
      int g = static_cast<int>(color.g);
      int b = static_cast<int>(color.b);
      int a = static_cast<int>(color.a);
    }
    vert_idx++;
  }
  LOG(INFO)<<"Draw Current TSDF";
  aru::core::utilities::viewer::Viewer::ViewVoxelPointCloud(
      voxel_points_, color_points_, position_vector_);
}

//------------------------------------------------------------------------------
void MeshMap::UpdateMap(utilities::image::StereoImage image_new) {

  cv::Mat image_left_grey, image_right_grey;
  cv::cvtColor(image_new.first.GetImage(), image_left_grey, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_new.second.GetImage(), image_right_grey,
               cv::COLOR_BGR2GRAY);
  // Match the features between the current frame and the new image
  //  viso_extractor_->FeaturesUntracked(image_left_grey, image_right_grey);
  //  frame_features_ = viso_extractor_->GetCurrentFeatures();

  frame_features_ =
      orb_matcher_->ComputeStereoMatches(image_left_grey, image_right_grey);

  VLOG(2) << "Number of features is " << frame_features_->size();

  viewer_->ViewImageFeatures(image_new.first.GetImage(), frame_features_);

  // Calculate the pose
  UpdateFramePose();
  VLOG(2) << "Current pose is  \n" << curr_pose_.GetTransform().matrix();

  // Update the position
  current_position_ = current_position_ * curr_pose_.GetTransform().matrix();
  position_vector_.emplace_back(current_position_);
  VLOG(2) << "Current position is  \n" << current_position_.matrix();
  LOG(INFO) << "Frame number " << frame_no;

  // viewer_->ViewPoseChain(position_vector_);
  //  Start a new frame
  frame_no++;
  // Update the current frame
  curr_frame_ =
      image::StereoImage{image::Image(frame_no, image_new.first.GetImage()),
                         image::Image(frame_no, image_new.second.GetImage())};

  // Calculate the mesh at this frame
  UpdateFrameMesh();
}
//------------------------------------------------------------------------------
void MeshMap::UpdateFramePose() {
  curr_pose_ = vo_->EstimateMotion(frame_features_);
  pose_chain_.push_back(curr_pose_);
}
} // namespace mesh_map
} // namespace mapping
} // namespace core
} // namespace aru
