//
// Created by paulamayo on 2022/04/14.
//

#ifndef ARU_CORE_MESH_MAP_SYSTEM_H
#define ARU_CORE_MESH_MAP_SYSTEM_H

#include <aru/core/mapping/mesh_mapping/laser_mesh_map.h>
#include <aru/core/mapping/mesh_mapping/mesh_map.h>
#include <aru/core/utilities/image/feature_tracker.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <aru/core/vo/vo.h>
#include <pbStereoImage.pb.h>
#include <pbTransform.pb.h>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>
#include <pbLaser.pb.h>

namespace aru {
namespace core {
namespace mapping {
namespace mesh_map {

class StereoSystem {

public:
  StereoSystem(std::string mapping_config_file,
               std::string image_left_monolithic,
               std::string image_right_monolithic, std::string output_ply);

  void Run();

  ~StereoSystem() = default;

private:
  std::string output_ply_;
  boost::shared_ptr<MeshMap> mesh_mapper_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_left_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_right_logger_;
};

class DepthSystem {
public:
  DepthSystem(std::string mapping_config_file, std::string image_rgb_monolithic,
              std::string image_depth_monolithic, std::string vo_monolithic,
              std::string output_ply);

  void Run();
  void RunPosesOnly();

  ~DepthSystem() = default;

private:
  std::string output_ply_;
  boost::shared_ptr<MeshMap> mesh_mapper_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_rgb_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_depth_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
      vo_logger_;
};

class LaserSystem {

public:
  LaserSystem(std::string mapping_config_file, std::string image_rgb_monolithic,
              std::string laser_monolithic, std::string vo_monolithic,
              std::string output_ply);

  void Run();

  ~LaserSystem() = default;

private:
  std::string output_ply_;
  boost::shared_ptr<LaserMeshMap> laser_mesh_mapper_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_rgb_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::laser::pbLaser>>
      laser_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
      vo_logger_;
};
} // namespace mesh_map
} // namespace mapping
} // namespace core
} // namespace aru

#endif // ARU_CORE_VO_SYSTEM_H
