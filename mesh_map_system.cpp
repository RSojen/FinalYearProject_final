//
// Created by paulamayo on 2022/04/14.
//

#include "aru/core/mapping/mesh_mapping/mesh_map_system.h"
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/laser/laserprotocolbufferadaptor.h>
#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <boost/make_shared.hpp>
#include <utility>
using namespace datatype::image;
using namespace datatype::transform;
using namespace datatype::laser;
using namespace aru::core::utilities;

namespace aru {
namespace core {
namespace mapping {
namespace mesh_map {
//------------------------------------------------------------------------------
StereoSystem::StereoSystem(std::string mapping_config_file,
                           std::string image_left_monolithic,
                           std::string image_right_monolithic,
                           std::string output_ply)
    : output_ply_(std::move(output_ply)) {

  mesh_mapper_ = boost::make_shared<MeshMap>(mapping_config_file);
  image_left_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_left_monolithic, false);
  image_right_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_right_monolithic, false);
}
//------------------------------------------------------------------------------
void StereoSystem::Run() {

  // Read previous image
  pbImage image_left_prev = image_left_logger_->ReadFromFile();
  image::Image prev_image_left =
      image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
          image_left_prev);
  pbImage image_right_prev = image_right_logger_->ReadFromFile();
  image::Image prev_image_right =
      image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
          image_right_prev);

  cv::Mat image_1_left_grey, image_1_right_grey;

  mesh_mapper_->InitialiseMap(
      image::StereoImage(prev_image_left, prev_image_right));

  pbImage image_left_curr = image_left_logger_->ReadFromFile();
  pbImage image_right_curr = image_right_logger_->ReadFromFile();
  int num = 0;
  while (!image_left_logger_->EndOfFile() &&
         !image_right_logger_->EndOfFile()) {

    // Perform the estimation
    auto estimation_start = std::chrono::high_resolution_clock::now();
    image::Image curr_image_left =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_left_curr);

    image::Image curr_image_right =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_curr);

    mesh_mapper_->UpdateMap(
        image::StereoImage(curr_image_left, curr_image_right));

    cv::cvtColor(curr_image_left.GetImage(), image_1_left_grey,
                 cv::COLOR_BGR2GRAY);
    cv::cvtColor(curr_image_right.GetImage(), image_1_right_grey,
                 cv::COLOR_BGR2GRAY);

    image_left_curr = image_left_logger_->ReadFromFile();
    image_right_curr = image_right_logger_->ReadFromFile();
  }
  LOG(INFO) << "Saving mesh to ply";
  //mesh_mapper_->SaveCurrentTsdf(output_ply_);
}

//------------------------------------------------------------------------------
DepthSystem::DepthSystem(std::string mapping_config_file,
                         std::string image_left_monolithic,
                         std::string image_depth_monolithic,
                         std::string vo_monolithic, std::string output_ply)
    : output_ply_(std::move(output_ply)) {

  mesh_mapper_ = boost::make_shared<MeshMap>(mapping_config_file);
  image_rgb_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_left_monolithic, false);
  image_depth_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_depth_monolithic, false);
  vo_logger_ = boost::make_shared<logging::ProtocolLogger<pbTransform>>(
      vo_monolithic, false);
}
//------------------------------------------------------------------------------
void DepthSystem::Run() {
  // Read the Vo monolithic into the mesh_map
  pbTransform pb_transform = vo_logger_->ReadFromFile();
  while (!vo_logger_->EndOfFile()) {
    transform::Transform curr_transform = aru::core::utilities::transform::
        TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            curr_transform);
    mesh_mapper_->ReadTransform(curr_transform_sptr);
    pb_transform = vo_logger_->ReadFromFile();
  }
  pbImage image_rgb_curr = image_rgb_logger_->ReadFromFile();
  pbImage image_depth_curr = image_depth_logger_->ReadFromFile();

  //filepath to save deformation graph data for plotting and the voxel size to be used for the mesh downsampling
  std::string dfg_unopt_file_path = "/home/ritvik/FinalYearProject/src/PGMO/plots/unoptimised_zoo_long.dgrf";
  std::string dfg_opt_file_path = "/home/ritvik/FinalYearProject/src/PGMO/plots/optimised_zoo_long.dgrf";

  //create file objects
  std::ofstream unopt(dfg_unopt_file_path);
  std::ofstream opt(dfg_opt_file_path);
  double voxelSize = 4;
  double octreeVoxelSize=0.1;

  //create octrees
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr full(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(octreeVoxelSize));
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr simple(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(octreeVoxelSize));

  //create noise model for poses
  auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << gtsam::Vector3::Constant(0.01), gtsam::Vector3::Constant(0.03))
          .finished());
  //noise model for loop closure
  auto loopNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << gtsam::Vector3::Constant(0.001), gtsam::Vector3::Constant(0.003))
          .finished());
  //create noise model for mesh Nodes
  auto meshToPoseNoise = gtsam::noiseModel::Isotropic::Variance(3, 0.1);
  auto meshToMeshNoise = gtsam::noiseModel::Isotropic::Variance(3, 1);
  //create cumulative cloud pointer
  pcl::PointCloud<pcl::PointXYZ>::Ptr cumulative_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  //create simplified cloud pointer
  pcl::PointCloud<pcl::PointXYZ>::Ptr simplified_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  //create Factor graph pointer
  gtsam::NonlinearFactorGraph::shared_ptr def_graph(new gtsam::NonlinearFactorGraph);
  //create gtsam values pointer
  gtsam::Values::shared_ptr vals(new gtsam::Values);

  //initialize octrees
  full->setInputCloud(cumulative_cloud);
  simple->setInputCloud(simplified_cloud);


  //create deformation graph object  
  DeformationGraph graph(unopt, opt, octreeVoxelSize, poseNoise, meshToMeshNoise, meshToPoseNoise, loopNoise, cumulative_cloud,simplified_cloud,full,simple,def_graph,vals);

  //boolean variable to check if loop closed
  bool loop_closure=false;

  int num = 0;

  int poseIndex=0;

  long prev_timestamp_keyframe=0;

  //create block hash object
  //voxblox::AnyIndexHashMapType<std::vector<int> >::type block_map;
  auto integration_start = std::chrono::high_resolution_clock::now();

  while (!image_rgb_logger_->EndOfFile() && !image_depth_logger_->EndOfFile()) {

    // Perform the estimation
    auto estimation_start = std::chrono::high_resolution_clock::now();
    image::Image curr_image_rgb =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_rgb_curr);

    image::Image curr_image_depth =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_depth_curr);


    //extract current timestamp
    long curr_timeStamp = curr_image_depth.GetTimeStamp();
    if (num%1==0||num==200)
    {
        mesh_mapper_->InsertDepthImage(curr_image_depth, curr_image_rgb);
        //LOG(INFO) << "inserted depth image";
        //get transform map pointer
        boost::shared_ptr<utilities::transform::TransformMap> transform_ptr = mesh_mapper_->getTransformMapPtr();

        utilities::transform::TransformSPtr global_position = transform_ptr->Interpolate(curr_timeStamp);

        if(global_position){
            //LOG(INFO) << "pointer valid";
            gtsam::Rot3 global_rot(global_position->GetRotation().cast<double>());
            gtsam::Point3 global_trans(
                    global_position->GetTranslation().cast<double>());
            gtsam::Pose3 global_pose(global_rot, global_trans);
            //LOG(INFO) << "created global pose estimate";

            //add pose to the graph
            graph.AddPose(global_pose,poseIndex);
            //LOG(INFO) << "added global pose estimate to initialize the graph";
        }

        //check if this is the first pose, if so add a prior factor and set estimate to the origin
        if (num==0)
        {
            graph.addPrior(num);
        }
        else
        {
            // Check for loop closure
            /*auto loop_closure = localiser_->FindLoopClosure(
                    curr_image_rgb, num - max_inter_frames_);
            LOG(INFO) << "Image " << num << " localised to image "
                      << loop_closure.first << " with probability "
                      << loop_closure.second;
            if (loop_closure.first > 0) {
                utilities::transform::Transform pose =
                        vo_->EstimateMotion(stereo_vector[loop_closure.first], image_frame);

                if (pose.GetTranslation().norm() > 0 &&
                    pose.GetTranslation().norm() < 2) {
                    LOG(INFO) << "Transform is " << pose.GetTransform().matrix();
                    gtsam::Rot3 loop_rot(pose.GetRotation().cast<double>());
                    gtsam::Point3 loop_trans(pose.GetTranslation().cast<double>());
                    gtsam::Pose3 loop_pose(loop_rot, loop_trans);
                    graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                            loop_closure.first, num_cameras_, curr_pose, poseNoise);

                    cv::Mat localised;
                    cv::hconcat(image_curr,
                                stereo_vector[loop_closure.first].first.GetImage(),
                                localised);

                    cv::resize(localised, localised, cv::Size(), 0.5, 0.5);
                    cv::imshow("Localisation", localised);
                    cv::waitKey(0);
                }
            }*/
            if(num==200)
            {
                loop_closure=true;
                graph.AddFakeLoopClosure(poseIndex,0);
            }
            //add odometry
            utilities::transform::TransformSPtr curr_position=transform_ptr->Interpolate(prev_timestamp_keyframe,curr_timeStamp);

            if (curr_position) {
                Eigen::Vector3f xyz = curr_position->GetTranslation();
                gtsam::Rot3 curr_rot(curr_position->GetRotation().cast<double>());
                gtsam::Point3 curr_trans(xyz.cast<double>());
                gtsam::Pose3 rel_pose(curr_rot, curr_trans);

                graph.AddOdomEdge(poseIndex-1,poseIndex,rel_pose);
            }

        }

        mesh_mapper_->BuildDfgFromTsdf(graph,poseIndex,voxelSize,loop_closure);
        //increment poseIndex
        poseIndex++;
        //set the previous timestamp
        prev_timestamp_keyframe=curr_timeStamp;

    }


    //mesh_mapper_->DrawCurrentTsdf();

    image_rgb_curr = image_rgb_logger_->ReadFromFile();
    image_depth_curr = image_depth_logger_->ReadFromFile();
    //LOG(INFO) << "Number is " << num;

    num++;

    if(loop_closure==true)
    {
        break;
    }

  }
  graph.addMeshValences();
  auto integration_end = std::chrono::high_resolution_clock::now();
  auto integration_time = std::chrono::duration_cast<std::chrono::seconds>(integration_end - integration_start);
  LOG(INFO) << "integrating depth frames and building the graph takes: " << integration_time.count() << " seconds";

  LOG(INFO) << "starting optimisation...";

  auto optimisation_start = std::chrono::high_resolution_clock::now();
  graph.optimiseLM();
  auto optimisation_end = std::chrono::high_resolution_clock::now();
  auto optimisation_time = std::chrono::duration_cast<std::chrono::seconds>(optimisation_end - optimisation_start);
  LOG(INFO) << "optimising the graph takes: " << optimisation_time.count() << " seconds";

  graph.closeDfgFiles();
  LOG(INFO) << "Finished LM";
  LOG(INFO) << "Saving optimised mesh to ply...";

  //unoptimised mesh file path
  std::string unopt_mesh = "/home/ritvik/FinalYearProject/src/PGMO/plots/unoptimised_zoo_long.ply";
  //optimised mesh file path
  std::string opt_mesh = "/home/ritvik/FinalYearProject/src/PGMO/plots/optimised_zoo_long.ply";

  auto mesh_start = std::chrono::high_resolution_clock::now();

  mesh_mapper_->PoseAndMeshSaveCurrentTsdf(unopt_mesh,opt_mesh,graph,simplified_cloud);

  auto mesh_end = std::chrono::high_resolution_clock::now();

  auto mesh_time = std::chrono::duration_cast<std::chrono::seconds>(mesh_end - mesh_start);

  LOG(INFO) << "deforming the mesh takes: " << mesh_time.count() << " seconds";


}
//------------------------------------------------------------------------------
void DepthSystem::RunPosesOnly() {
        // Read the Vo monolithic into the mesh_map
        pbTransform pb_transform = vo_logger_->ReadFromFile();
        while (!vo_logger_->EndOfFile()) {
            transform::Transform curr_transform = aru::core::utilities::transform::
            TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
            aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
                    boost::make_shared<aru::core::utilities::transform::Transform>(
                            curr_transform);
            mesh_mapper_->ReadTransform(curr_transform_sptr);
            pb_transform = vo_logger_->ReadFromFile();
        }
        pbImage image_rgb_curr = image_rgb_logger_->ReadFromFile();
        pbImage image_depth_curr = image_depth_logger_->ReadFromFile();

        //filepath to save deformation graph data for plotting and the voxel size to be used for the mesh downsampling
        std::string dfg_unopt_file_path = "/home/ritvik/FinalYearProject/src/PGMO/plots/unoptimised_poses_zoo.dgrf";
        std::string dfg_opt_file_path = "/home/ritvik/FinalYearProject/src/PGMO/plots/optimised_poses_zoo.dgrf";

        //create file objects
        std::ofstream unopt(dfg_unopt_file_path);
        std::ofstream opt(dfg_opt_file_path);

        double octreeVoxelSize = 0.1;

        auto loopNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << gtsam::Vector3::Constant(0.001), gtsam::Vector3::Constant(0.003))
                        .finished());

        //create octrees
        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr full(
                new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(octreeVoxelSize));
        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr simple(
                new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(octreeVoxelSize));

        //create noise model for poses
        auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << gtsam::Vector3::Constant(0.01), gtsam::Vector3::Constant(0.03))
                        .finished());
        //create noise model for mesh Nodes
        auto meshNoise = gtsam::noiseModel::Isotropic::Variance(3, 1e-3);
        //create cumulative cloud pointer
        pcl::PointCloud<pcl::PointXYZ>::Ptr cumulative_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //create simplified cloud pointer
        pcl::PointCloud<pcl::PointXYZ>::Ptr simplified_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //create Factor graph pointer
        gtsam::NonlinearFactorGraph::shared_ptr def_graph(new gtsam::NonlinearFactorGraph);
        //create gtsam values pointer
        gtsam::Values::shared_ptr vals(new gtsam::Values);

        //initialize octrees
        full->setInputCloud(cumulative_cloud);
        simple->setInputCloud(simplified_cloud);

        long prev_timestamp = 0;
        //create deformation graph object
        DeformationGraph graph(unopt, opt, octreeVoxelSize, poseNoise, meshNoise, meshNoise, loopNoise, cumulative_cloud, simplified_cloud,
                               full, simple, def_graph, vals);

        //boolean variable to check if loop closed
        bool loop_closure = false;

        int num = 0;

        int poseIndex=0;

        long prev_timestamp_keyframe=0;

        std::vector<image::Image> rgb;
        std::vector<image::Image> depth;
        std::vector<Eigen::Affine3f> poses;

        auto reading_start = std::chrono::high_resolution_clock::now();
        while (!image_rgb_logger_->EndOfFile() && !image_depth_logger_->EndOfFile()) {

            //Perform the estimation
            auto estimation_start = std::chrono::high_resolution_clock::now();
            image::Image curr_image_rgb =
                    image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
                            image_rgb_curr);

            image::Image curr_image_depth =
                    image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
                            image_depth_curr);



            //display image
            //cv::Mat rgb_image = curr_image_rgb.GetImage();
            //cv::imshow("rgb image",rgb_image);
            //cv::waitKey(5000000);

            //display depth map
            //cv::Mat depth_map = curr_image_depth.GetImage();
            //cv::imshow("depth map",depth_map);
            //cv::waitKey(5000000);



            //extract current timestamp
            long curr_timeStamp = curr_image_depth.GetTimeStamp();

            if (num%1==0||num==200)
            {
                rgb.push_back(curr_image_rgb);
                depth.push_back(curr_image_depth);

                //get transform map pointer
                boost::shared_ptr<utilities::transform::TransformMap> transform_ptr = mesh_mapper_->getTransformMapPtr();

                utilities::transform::TransformSPtr global_position = transform_ptr->Interpolate(curr_timeStamp);

                if (global_position) {

                    gtsam::Rot3 global_rot(global_position->GetRotation().cast<double>());
                    gtsam::Point3 global_trans(
                            global_position->GetTranslation().cast<double>());
                    gtsam::Pose3 global_pose(global_rot, global_trans);


                    //add pose to the graph
                    graph.AddPose(global_pose, poseIndex);

                }

                //check if this is the first pose, if so add a prior factor and set estimate to the origin
                if (poseIndex == 0) {
                    graph.addPrior(poseIndex);
                } else {

                    if (num == 200) {
                        loop_closure = true;
                        graph.AddFakeLoopClosure(poseIndex, 0);
                    }
                    //add odometry
                    utilities::transform::TransformSPtr curr_position = transform_ptr->Interpolate(prev_timestamp_keyframe,
                                                                                                   curr_timeStamp);

                    if (curr_position) {
                        Eigen::Vector3f xyz = curr_position->GetTranslation();
                        gtsam::Rot3 curr_rot(curr_position->GetRotation().cast<double>());
                        gtsam::Point3 curr_trans(xyz.cast<double>());
                        gtsam::Pose3 rel_pose(curr_rot, curr_trans);

                        graph.AddOdomEdge(poseIndex - 1, poseIndex, rel_pose);
                    }

                }
                //increment poseIndex
                poseIndex++;
                //set the previous timestamp
                prev_timestamp_keyframe=curr_timeStamp;
            }


            image_rgb_curr = image_rgb_logger_->ReadFromFile();
            image_depth_curr = image_depth_logger_->ReadFromFile();


            prev_timestamp = curr_timeStamp;
            num++;

            if (loop_closure == true) {
                break;
            }

        }
        auto reading_end = std::chrono::high_resolution_clock::now();

        auto reading_time = std::chrono::duration_cast<std::chrono::seconds>(reading_end - reading_start);

        LOG(INFO) << "reading takes " << reading_time.count() << " seconds";

        LOG(INFO) << "starting optimisation...";

        auto optimisation_start = std::chrono::high_resolution_clock::now();
        graph.optimiseLM();
        auto optimisation_end = std::chrono::high_resolution_clock::now();

        auto optimisation_time = std::chrono::duration_cast<std::chrono::microseconds>(optimisation_end - optimisation_start);

        LOG(INFO) << "Optimisation takes " << optimisation_time.count() << " microseconds";

        LOG(INFO) << "Finished LM";

        graph.closeDfgFiles();

        graph.extractPoses(poses);

        auto depth_map_integration_start = std::chrono::high_resolution_clock::now();
        //construct the map
        for(int i=1;i<poses.size();++i)
        {
            cv::Mat disparity = depth[i].GetImage();
            cv::Mat rgb_image = rgb[i].GetImage();
            cv::Mat depth = mesh_mapper_->getMeshPtr()->DisparityToDepth(disparity);
            mesh_mapper_->InsertDepthImage(depth,rgb_image,poses[i]);
        }
        auto depth_map_integration_end = std::chrono::high_resolution_clock::now();
        auto depth_integration_time = std::chrono::duration_cast<std::chrono::seconds>(depth_map_integration_end - depth_map_integration_start);

        LOG(INFO) << "Depth Integration takes " << depth_integration_time.count() << " seconds";

        LOG(INFO) << "Saving optimised mesh to ply...";

        //optimised mesh file path
        std::string opt_mesh = "/home/ritvik/FinalYearProject/src/PGMO/plots/zoo_poses_only.ply";


        auto mesh_start = std::chrono::high_resolution_clock::now();
        mesh_mapper_->SaveCurrentTsdf(opt_mesh);
        auto mesh_end = std::chrono::high_resolution_clock::now();

        auto mesh_writing_time = std::chrono::duration_cast<std::chrono::seconds>(mesh_end - mesh_start);

        LOG(INFO) << "Mesh writing takes " << mesh_writing_time.count() << " seconds";

}
//------------------------------------------------------------------------------
LaserSystem::LaserSystem(std::string mapping_config_file,
                         std::string image_rgb_monolithic,
                         std::string laser_monolithic,
                         std::string vo_monolithic, std::string output_ply)
    : output_ply_(std::move(output_ply)) {

  laser_mesh_mapper_ = boost::make_shared<LaserMeshMap>(mapping_config_file);
  image_rgb_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_rgb_monolithic, false);
  laser_logger_ = boost::make_shared<logging::ProtocolLogger<pbLaser>>(
      laser_monolithic, false);
  vo_logger_ = boost::make_shared<logging::ProtocolLogger<pbTransform>>(
      vo_monolithic, false);
}
//------------------------------------------------------------------------------
void LaserSystem::Run() {
  // Read the Vo monolithic into the mesh_map
  pbTransform pb_transform = vo_logger_->ReadFromFile();
  while (!vo_logger_->EndOfFile()) {
    transform::Transform curr_transform = aru::core::utilities::transform ::
        TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            curr_transform);
    laser_mesh_mapper_->ReadTransform(curr_transform_sptr);
    pb_transform = vo_logger_->ReadFromFile();
  }

  image_rgb_logger_->ReadFromFile();
  // Read the following images
  LOG(INFO) << "Updating the map";
  pbLaser pb_laser = laser_logger_->ReadFromFile();
  pbImage pb_image_left = image_rgb_logger_->ReadFromFile();
  //for (int i = 0; i < 90; ++i) { image_rgb_logger_->ReadFromFile();}
  while (!image_rgb_logger_->EndOfFile()) {
  //for (int i = 0; i < 100; ++i) {
    image::Image curr_image =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            pb_image_left);
    laser::Laser curr_laser =
        laser::LaserProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_laser);
    laser_mesh_mapper_->UpdateMap(curr_image, curr_laser);
    while (curr_image.GetTimeStamp() > pb_laser.timestamp() &&
           !laser_logger_->EndOfFile()) {
      laser::Laser curr_laser =
          laser::LaserProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_laser);
      //      laser_mesh_mapper_->UpdateMap(curr_image, curr_laser);
      pb_laser = laser_logger_->ReadFromFile();
    }
    // pb_laser = laser_logger_->ReadFromFile();
    pb_image_left = image_rgb_logger_->ReadFromFile();

   // if(i%10==0) laser_mesh_mapper_->DrawCurrentTsdf();
  }

  laser_mesh_mapper_->SaveCurrentTsdf(output_ply_);
}

} // namespace mesh_map
} // namespace mapping
} // namespace core
} // namespace aru