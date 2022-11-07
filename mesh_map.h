#ifndef ARU_CORE_MAPPING_MESH_MAP_H_
#define ARU_CORE_MAPPING_MESH_MAP_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <string>

#include "aru/core/utilities/image/feature_tracker.h"
#include "aru/core/utilities/image/image.h"
#include "voxblox/core/tsdf_map.h"
#include "voxblox/integrator/tsdf_integrator.h"
#include "voxblox/mesh/mesh.h"
#include "voxblox/mesh/mesh_integrator.h"
#include "voxblox/mesh/mesh_layer.h"
#include <aru/core/mesh/mesh.h>
#include <aru/core/utilities/laser/laser.h>
#include <aru/core/utilities/transforms/transform_map.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <aru/core/vo/vo.h>
#include <opencv2/opencv.hpp>
#include <voxblox/alignment/icp.h>

#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/filters/voxel_grid.h>


#include <Eigen/Dense>
#include <Eigen/Core>

#define SLOW_BUT_CORRECT_BETWEENFACTOR

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>




namespace aru {
namespace core {
namespace mapping {
namespace mesh_map {


//The deformation edge factor is taken from https://github.com/MIT-SPARK/Kimera-PGMO/blob/master/include/kimera_pgmo/DeformationGraph.h
class DeformationEdge : public gtsam::NoiseModelFactor2<gtsam::Pose3,gtsam::Pose3>{
private:
    gtsam::Pose3 node1_pose;
    gtsam::Point3 node2_position;

public:
    DeformationEdge(gtsam::Symbol node1, gtsam::Symbol node2, gtsam::Pose3 node1_pose, gtsam::Point3 node2_point, gtsam::SharedNoiseModel noise_model) :
    gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(noise_model, node1, node2), node1_pose(node1_pose), node2_position(node2_point){}
    ~DeformationEdge() {}
    //error function
    gtsam::Vector evaluateError(
            const gtsam::Pose3& p1,
            const gtsam::Pose3& p2,
            boost::optional<gtsam::Matrix&> H1 = boost::none,
            boost::optional<gtsam::Matrix&> H2 = boost::none) const{
        // position of node 2 in frame of node 1
        gtsam::Point3 t_12 = node1_pose.rotation().inverse().rotate(
        node2_position - node1_pose.translation());

        gtsam::Matrix H_R1, H_t1, H_t2;
        gtsam::Rot3 R1 = p1.rotation();
        gtsam::Point3 t1 = p1.translation(H_t1);
        // New position of node 2 according to deformation p1 of node 1
        gtsam::Point3 t2_1 = t1 + R1.rotate(t_12, H_R1);
        gtsam::Point3 t2_2 = p2.translation(H_t2);

        // Calculate Jacobians
        Eigen::MatrixXd Jacobian_1 = Eigen::MatrixXd::Zero(3, 6);
        Jacobian_1.block<3, 3>(0, 0) = H_R1;
        Jacobian_1 = Jacobian_1 + H_t1;
        Eigen::MatrixXd Jacobian_2 = Eigen::MatrixXd::Zero(3, 6);
        Jacobian_2 = Jacobian_2 - H_t2;

        if (H1) *H1 = Jacobian_1;
        if (H2) *H2 = Jacobian_2;

        return t2_1 - t2_2;
    }
};
class DeformationGraph{
private:
    //gtsam structure to hold deformation graph connectivity
    gtsam::NonlinearFactorGraph::shared_ptr graph;
    //gtsam structure to hold values in deformation graph
    gtsam::Values::shared_ptr def_graph_vals;
    //structure to hold optimized values
    gtsam::Values::shared_ptr opt_vals;
    //file to write deformation graph data to for further plotting
    std::ofstream& dfg_file_unopt_out;
    std::ofstream& dfg_file_opt_out;
    //define type for pcl pointcloud pointer
    typedef pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_ptr;
    //mapping between mesh vertex and poses it has been observed by
    std::unordered_map<int, std::set<int> > VertexToPose;
    //mapping between simplified deformation graph node index and poses it has been observed by
    std::unordered_map<int, std::set<int> > NodeToPose;
    //voxelgrid filter to merge vertices in the same voxel
    pcl::VoxelGrid<pcl::PointXYZ> simple;
    //cumulative cloud
    pointcloud_ptr cum_cloud;
    //simplified cloud
    pointcloud_ptr simp_cloud;
    //noise model for the pose-to-pose measurements
    gtsam::SharedNoiseModel poseNoise;
    //noise model for the mesh nodes
    gtsam::SharedNoiseModel meshToMeshNoise;
    gtsam::SharedNoiseModel meshToPoseNoise;
    //loop closure noise
    gtsam::SharedNoiseModel loopNoise;
    //voxel dimension
    double voxel_dim;
    typedef pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr octree_ptr;
    //octree for the full pointcloud
    octree_ptr octree_full;
    //octree for the simplified pointcloud
    octree_ptr octree_simp;
    

public:
    DeformationGraph(std::ofstream& unopt, std::ofstream& opt, double& voxelSize,gtsam::SharedNoiseModel PosenoiseModel,gtsam::SharedNoiseModel MeshToMeshNoiseModel, gtsam::SharedNoiseModel MeshToPoseNoiseModel, gtsam::SharedNoiseModel loopClosureNoise,pointcloud_ptr cloud1, pointcloud_ptr cloud2, octree_ptr octree_cum, octree_ptr octree_simplified, gtsam::NonlinearFactorGraph::shared_ptr def_graph, gtsam::Values::shared_ptr vals) : voxel_dim(voxelSize), poseNoise(PosenoiseModel), meshToMeshNoise(MeshToMeshNoiseModel), meshToPoseNoise(MeshToPoseNoiseModel), loopNoise(loopClosureNoise), cum_cloud(cloud1), simp_cloud(cloud2),
    def_graph_vals(vals), graph(def_graph), dfg_file_unopt_out(unopt), dfg_file_opt_out(opt), opt_vals(new gtsam::Values), octree_full(octree_cum), octree_simp(octree_simplified)
    {

        //check if file could be opened
        if(dfg_file_unopt_out&&dfg_file_opt_out){
          LOG(INFO) << "files opened successfully";
        }
        else{
          LOG(INFO) << "could not open files";
        }
    }
    //function to close dfg files
    void closeDfgFiles()
    {
        dfg_file_unopt_out.close();
        dfg_file_opt_out.close();
    }
    //function to fuse Node into current map
    void FuseMeshNode(pcl::PointXYZ node, int pose, std::vector<pcl::PointXYZ>& points, double mergeDist)
    {
        if(pose==1)
        {
            octree_simp->addPointToCloud(node,simp_cloud);
            std::set<int> poses;
            poses.insert(1);

            //insert at end index
            int index = simp_cloud->size()-1;
            NodeToPose.insert(std::pair<int, std::set<int> >(index,poses));
            //add to initial values
            this->AddMeshNode(node,index);

            //add to deformation graph
            this->AddPoseMeshEdge(pose,index);

        }
        else{
            //set octree resolution to voxel size
            //double resolution = voxel_dim;

            //add simplified pointcloud to octree data structure
            //pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
            //octree.setInputCloud(simp_cloud);
            //octree.addPointsFromInputCloud();

            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;

            octree_simp->nearestKSearch(node, 1, pointIdxNKNSearch, pointNKNSquaredDistance);

            int closest_index = pointIdxNKNSearch[0];
            float closest_distance = pointNKNSquaredDistance[0];
            //check if nodes have overlapped or are very close
            int earliest_pose = *(NodeToPose.at(closest_index).begin());
            //LOG(INFO) << "earliest pose: " << earliest_pose;
            //LOG(INFO) << "current pose: " << pose;
            if (closest_distance < mergeDist)
            {
                if (abs(pose-earliest_pose)<2)
                {
                  //LOG(INFO) << "merged";
                  //merge the nodes
                  NodeToPose.at(closest_index).insert(pose);
                  //add the edge to the deformation graph
                  this->AddPoseMeshEdge(pose,closest_index);
                }
                else{
                  //create a new node
                  points.push_back(node);
                  std::set<int> poses;
                  poses.insert(pose);

                  //insert at end index
                  int index = simp_cloud->size()+(points.size()-1);
                  NodeToPose.insert(std::pair<int, std::set<int> >(index,poses));
                  //add to initial values
                  this->AddMeshNode(node,index);

                  //add to deformation graph
                  this->AddPoseMeshEdge(pose,index);
                }
            }
            else{
                //create a new node
                //LOG(INFO) << "not merged";
                points.push_back(node);
                std::set<int> poses;
                poses.insert(pose);

                //insert at end index
                int index = simp_cloud->size()+(points.size()-1);
                NodeToPose.insert(std::pair<int, std::set<int> >(index,poses));
                //add to initial values
                this->AddMeshNode(node,index);

                //add to deformation graph
                this->AddPoseMeshEdge(pose,index);

            }

        }
    }
    //function to fuse vertex into current map
    void FuseMeshVertex(pcl::PointXYZ node, int pose, std::vector<pcl::PointXYZ>& points)
    {
        if(pose==1)
        {
            octree_full->addPointToCloud(node,cum_cloud);
            std::set<int> poses;
            poses.insert(1);

            //insert at end index
            int index = cum_cloud->size()-1;
            VertexToPose.insert(std::pair<int, std::set<int> >(index,poses));

        }
        else{
            //set octree resolution to voxel size
            //double resolution = voxel_dim;

            //add simplified pointcloud to octree data structure
            //pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
            //octree.setInputCloud(simp_cloud);
            //octree.addPointsFromInputCloud();

            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;

            octree_full->nearestKSearch(node, 1, pointIdxNKNSearch, pointNKNSquaredDistance);

            int closest_index = pointIdxNKNSearch[0];
            float closest_distance = pointNKNSquaredDistance[0];
            //check if nodes have overlapped or are very close
            if (closest_distance < 0.1)
            {
                int earliest_pose = *(VertexToPose.at(closest_index).begin());
                if (abs(pose-earliest_pose)<2)
                {
                  //merge the nodes
                  VertexToPose.at(closest_index).insert(pose);
                }
                else{
                  //create a new node
                  points.push_back(node);
                  std::set<int> poses;
                  poses.insert(pose);

                  //insert at end index
                  int index = cum_cloud->size()+(points.size()-1);
                  VertexToPose.insert(std::pair<int, std::set<int> >(index,poses));
                  
                }
            }
            else{
                //create a new node
                points.push_back(node);
                std::set<int> poses;
                poses.insert(pose);

                //insert at end index
                int index = cum_cloud->size()+(points.size()-1);
                VertexToPose.insert(std::pair<int, std::set<int> >(index,poses));
            }

        }
    }
  
  
    //function to add odometry edge to the deformation graph
    void AddOdomEdge(int pose_t,int pose_t1, gtsam::Pose3 rel_transform)
    {
      gtsam::Symbol pose_t_prev('x',pose_t);
      gtsam::Symbol pose_t_next('x',pose_t1);
      graph->add(gtsam::BetweenFactor<gtsam::Pose3>(pose_t_prev, pose_t_next, rel_transform, poseNoise));

      //write to file
      dfg_file_opt_out<< "BETWEEN " << 'x' << pose_t << ' ' << 'x' << pose_t1 << std::endl;
      dfg_file_unopt_out<< "BETWEEN " << 'x' << pose_t << ' ' << 'x' << pose_t1 << std::endl;

    }
    //add prior factor to the first pose - this ties the graph to the origin
    void addPrior(int poseIndex)
    {
      gtsam::Symbol poseSym('x',poseIndex);
      gtsam::Pose3 origin(gtsam::Rot3(), gtsam::Point3(0,0,0));
      //add pose estimate
      def_graph_vals->insert(poseSym,origin);
      dfg_file_unopt_out << "NODE " << poseSym.chr() << poseSym.index() << " " << 0 << " " << 0 << " " << 0 << std::endl;
      //add factor
      graph->addPrior(poseSym, origin, poseNoise);
    }
    //add pose to initial estimate
    void AddPose(gtsam::Pose3& pose, int poseIndex)
    {
      gtsam::Symbol poseSym('x',poseIndex);
      def_graph_vals->insert(poseSym, pose);
      dfg_file_unopt_out << "NODE " << poseSym.chr() << poseSym.index() << " " << pose.translation().x() << " " << pose.translation().y() << " " << pose.translation().z() << std::endl;
    }
    //Add false Loop Closing factor - set the transformation to identity to assume we have observed exactly the same place
    void AddFakeLoopClosure(int pose1_index, int pose2_index)
    {
      gtsam::Symbol pose1('x',pose1_index);
      gtsam::Symbol pose2('x',pose2_index);
      gtsam::Pose3 rel_transform(gtsam::Rot3(),gtsam::Point3(0,0,0));
      //add loop closure
      graph->add(gtsam::BetweenFactor<gtsam::Pose3>(pose1, pose2, rel_transform, loopNoise));
      //add to files
      dfg_file_opt_out<< "BETWEEN " << 'x' << pose1_index << ' ' << 'x' << pose2_index << std::endl;
      dfg_file_unopt_out<< "BETWEEN " << 'x' << pose1_index << ' ' << 'x' << pose2_index << std::endl;
    }
    //Add mesh nodes to initial values
    void AddMeshNode(pcl::PointXYZ point, int index)
    {
        gtsam::Symbol vertex('v',index);
        float x = point.x;
        float y = point.y;
        float z = point.z;
        gtsam::Point3 vertex_pos(x,y,z);
        //add to initial estimate
        def_graph_vals->insert(vertex,gtsam::Pose3(gtsam::Rot3(),vertex_pos));
        dfg_file_unopt_out << "NODE " << 'v' << index << " " << x << " " << y << " " << z << std::endl;
    }
    //add deformation edges between pose nodes and observed mesh nodes
    void AddPoseMeshEdge(int pose, int node_idx)
    {
      
          gtsam::Symbol poseNode('x',pose);
          gtsam::Symbol MeshNode('v',node_idx);
          const DeformationEdge edge(poseNode, MeshNode, def_graph_vals->at<gtsam::Pose3>(poseNode),def_graph_vals->at<gtsam::Pose3>(MeshNode).translation(),meshToPoseNoise);
          const DeformationEdge edge2(MeshNode, poseNode, def_graph_vals->at<gtsam::Pose3>(MeshNode),def_graph_vals->at<gtsam::Pose3>(poseNode).translation(),meshToPoseNoise);
          graph->add(edge);
          graph->add(edge2);


          //add to files
          dfg_file_opt_out<< "DEDGE " << 'x' << pose << ' ' << 'v' << node_idx << std::endl;
          dfg_file_unopt_out<< "DEDGE " << 'x' << pose << ' ' << 'v' << node_idx << std::endl;
        
    }
    //add deformation edges between mesh nodes
    void addMeshValences()
    {


      //set octree resolution to voxel size
      //double resolution = voxel_dim;

      //add simplified pointcloud to octree data structure
      //pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
      //octree.setInputCloud(simp_cloud);
      //octree.addPointsFromInputCloud();

      //use a k-nearest neighbour approach to determine the connectivity of the deformation graph, we are using k=4 connectivity but set k to 6 in the case that some connections are invalid
      int K = 8;

      //store the index
      int index=0;

      //number of edges created
      int edges=0;

      //store a counter to check how many nodes have been connected
      int counter=0;

      for (pcl::PointCloud<pcl::PointXYZ>::iterator it = simp_cloud->begin(); it != simp_cloud->end(); ++it)
      {
        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;

        octree_simp->nearestKSearch(*it, K, pointIdxNKNSearch, pointNKNSquaredDistance);

        //LOG(INFO) << "numer of neighbours found: " << pointIdxNKNSearch.size();
        

        for(int connected_index : pointIdxNKNSearch)
        {

          int pose1 = *(NodeToPose.at(index).begin());
          int pose2 = *(NodeToPose.at(connected_index).begin());

          int distance = abs(pose1-pose2);

          if(distance < 5 && counter < 3 &&(index!=connected_index))
          {
            //create deformation edge between mesh nodes
            gtsam::Symbol v_from('v',index);
            gtsam::Symbol v_to('v',connected_index);
            const DeformationEdge edge(v_from, v_to, def_graph_vals->at<gtsam::Pose3>(v_from),def_graph_vals->at<gtsam::Pose3>(v_to).translation(),meshToMeshNoise);

            //add to files
            dfg_file_opt_out<< "DEDGE " << 'v' << index << ' ' << 'v' << connected_index << std::endl;
            dfg_file_unopt_out<< "DEDGE " << 'v' << index << ' ' << 'v' << connected_index << std::endl;

            //LOG(INFO) << 'v' << index << " connected to " << 'v' << connected_index;

            graph->add(edge);
            //LOG(INFO) << "added mesh edge to deformation graph";

            edges++;
            //LOG(INFO) << "edges: " << edges;

            counter++;
          }
          

        }
        index++;
        counter=0;

      }

    }
    //deform a vertex
    Eigen::Vector3d DeformVertex(pcl::PointXYZ vertex, int* overlap_counter)
    {
      
        //set octree resolution to voxel size
        //double resolution = voxel_dim;

        //search for closest vertex in fused map
        //pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
        //octree.setInputCloud(cum_cloud);
        //octree.addPointsFromInputCloud();
        //extract the vertex as a gtsam point
        gtsam::Point3 v(vertex.x,vertex.y,vertex.z);
        gtsam::Point3 v_new(0,0,0);

        //add simplified pointcloud to octree data structure
        //pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree_simp(resolution);
        //octree_simp.setInputCloud(simp_cloud);
        //octree_simp.addPointsFromInputCloud();

        //use a k-nearest neighbour approach to determine the connectivity of the deformation graph, we are using k=4 connectivity but set k to 6 in the case that some connections are invalid
        int K = 5;

        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;

        octree_simp->nearestKSearch(vertex, K, pointIdxNKNSearch, pointNKNSquaredDistance);

        //LOG(INFO) << "distances squared: [" << pointNKNSquaredDistance[0] << "," << pointNKNSquaredDistance[1] << "," << pointNKNSquaredDistance[2] << "," << pointNKNSquaredDistance[3] << "," << pointNKNSquaredDistance[4]<<"]";

        //store dmax index
        int dmax_index;

        //store valid node indices
        std::vector<int> valid_nodes;
        std::vector<int> valid_indices;


        int counter=0;

        bool connected=false;

        for(int i=0; i<pointIdxNKNSearch.size();++i)
        {
            if(counter<5)
            {
                if (counter==4){
                    dmax_index=i;
                    counter++;
                }
                else{
                    //add to valid nodes
                    valid_nodes.push_back(pointIdxNKNSearch[i]);
                    valid_indices.push_back(i);
                    counter++;
                }
            }
        }


        //extract d_max
        double d_max=sqrt(pointNKNSquaredDistance[dmax_index]);



        //weights
        std::vector<double> weights;

        //total weight
        double total_weight=0;

        //calculate weights
        for(int i=0; i<valid_indices.size();++i)
        {
            double weight=(1-(sqrt(pointNKNSquaredDistance[valid_indices[i]])/d_max));

            double new_weight=weight*weight;
            total_weight+=new_weight;
            weights.push_back(new_weight);
        }

        //normalise weights to sum to 1
        for(int i=0; i<weights.size();++i)
        {
            weights[i]=weights[i]/total_weight;
        }

        //calculate deformation
        for(int i=0; i<valid_nodes.size();++i)
        {
            int node_idx = valid_nodes[i];
            gtsam::Symbol node('v',node_idx);
            gtsam::Pose3 node_pose = opt_vals->at<gtsam::Pose3>(node);

            pcl::PointXYZ vertex_position = simp_cloud->at(node_idx);
            gtsam::Point3 vertex_pos(vertex_position.x,vertex_position.y,vertex_position.z);

            v_new += weights[i]*(node_pose.rotation().rotate(v-vertex_pos)+node_pose.translation());

        }

        Eigen::Vector3d out(v_new.x(),v_new.y(),v_new.z());

        return out;
    }
    //merge vertices in the same voxel and write to simplified pointcloud
    void mergeVertices(pointcloud_ptr simplified)
    {
      simple.setSaveLeafLayout(true);
      simple.setInputCloud(cum_cloud);
      simple.setLeafSize(voxel_dim,voxel_dim,voxel_dim);
      simple.filter(*simplified);
      LOG(INFO) << "simplified input point cloud";
    }
    //function to add point to cumulative cloud
    void addToCumCloud(pcl::PointXYZ& point)
    {
      octree_full->addPointToCloud(point,cum_cloud);
    }
    void addToSimpCloud(pcl::PointXYZ& point)
    {
        octree_simp->addPointToCloud(point,simp_cloud);
    }
    //function to retrieve current size of the point cloud;
    int getCloudSize()
    {
      return cum_cloud->size();
    }


    //function to map indices in simplified point cloud to connected pose indices
   /* void mapMeshNodeToPoses(pointcloud_ptr simplified_cloud)
    {
      int counter = 0;
      LOG(INFO) << "size of cloud: " << simplified_cloud->size();

      for (pcl::PointCloud<pcl::PointXYZ>::iterator point = simplified_cloud->begin(); point != simplified_cloud->end(); ++point)
      {
        std::set<int> poses;
        NodeToPose.insert(std::pair<int, std::set<int> >(counter, poses));
        counter++;
      }

      LOG(INFO) << "added all points from simplified cloud";

      counter = 0;

      for (pcl::PointCloud<pcl::PointXYZ>::iterator point = cum_cloud->begin(); point != cum_cloud->end(); ++point)
      {
        //get centroid index
        int centroid_index = simple.getCentroidIndex(*point);
        LOG(INFO) << "maps to centroid: " << centroid_index;
        //get connected poses and add to Node mapping
        for (auto pose : VertexToPose.at(counter))
        {
          LOG(INFO) << "maps to pose: " << pose;
          NodeToPose.at(centroid_index).insert(pose);
        }
        counter++;
      }
    }*/

    //optimise the graph and return values
    void optimiseLM()
    {
      gtsam::LevenbergMarquardtParams params;
      gtsam::LevenbergMarquardtParams::SetCeresDefaults(&params);
      params.setMaxIterations(100);
      params.setVerbosity("ERROR");
      gtsam::LevenbergMarquardtOptimizer opt(*graph, *def_graph_vals, params);
      *opt_vals = opt.optimize();

      //write to file
      for (auto val : opt_vals->keys())
      {
        gtsam::Symbol sym(val);
        dfg_file_opt_out << "NODE " << sym.chr() << sym.index() << " " << opt_vals->at<gtsam::Pose3>(sym).translation().x() << " " << opt_vals->at<gtsam::Pose3>(sym).translation().y() << " " << opt_vals->at<gtsam::Pose3>(sym).translation().z() << std::endl;
      }

    }
    //extract optimised poses
    void extractPoses(std::vector<Eigen::Affine3f>& poses)
    {
        for (auto val : opt_vals->keys())
        {
            gtsam::Symbol sym(val);
            gtsam::Pose3 pose = opt_vals->at<gtsam::Pose3>(sym);
            Eigen::Matrix3f rotation = pose.rotation().matrix().cast<float>();
            Eigen::Vector3f translation = pose.translation().matrix().cast<float>();

            //construct transformation matrix
            Eigen::Matrix4f transform;
            transform.setIdentity();
            transform.block<3,3>(0,0) = rotation;
            transform.block<3,1>(0,3) = translation;

            Eigen::Affine3f transformation(transform);

            poses.push_back(transformation);
        }
    }



};

class MeshMap {

public:
  explicit MeshMap(std::string mesh_map_settings_file);

  ~MeshMap() = default;

  void InitialiseMap(utilities::image::StereoImage image_init);

  void UpdateMap(utilities::image::StereoImage image_new);

  void ReadTransform(utilities::transform::TransformSPtr transform);

  void UpdateFrameMesh();

  void InsertDepthImage(utilities::image::Image depth_image,
                        utilities::image::Image rgb_image);

  void InsertDepthImage(cv::Mat depth_image, cv::Mat rgb_image,
                        Eigen::Affine3f position);

  void UpdateFramePose();

  void DrawCurrentTsdf();

  void SaveCurrentTsdf(std::string output_ply);

  void PoseAndMeshSaveCurrentTsdf(std::string output_ply_unopt, std::string output_ply_opt, DeformationGraph& graph, pcl::PointCloud<pcl::PointXYZ>::Ptr simplified);

  void BuildDfgFromTsdf(DeformationGraph& graph, int poseIndex, double voxelDim, bool loop_closed);

  boost::shared_ptr<utilities::transform::TransformMap> getTransformMapPtr(){return transform_map_;}
  boost::shared_ptr<mesh::Mesh> getMeshPtr(){return mesh_;}

  utilities::transform::Transform GetCurrentPose() { return curr_pose_; }

  std::vector<Eigen::Vector3d> GetCurrentPointCloud() {
    return curr_pointcloud_;
  }

  std::vector<Eigen::Vector3d> GetColorPointCloud() {
    return color_pointcloud_;
  }

private:
  std::string mesh_map_settings_file_;

  utilities::image::ExtractorParams extractor_params_;
  utilities::image::MatcherParams matcher_params_;
  vo::SolverParams solver_params_;

  mesh::RegularisationParams reg_params_;
  mesh::ViewerParams viewer_params_;
  utilities::camera::CameraParams camera_params_;

  utilities::image::FeatureSPtrVectorSptr map_features_;
  utilities::image::FeatureSPtrVectorSptr frame_features_;

  boost::shared_ptr<vo::VO> vo_;
  boost::shared_ptr<mesh::Mesh> mesh_;

  boost::shared_ptr<voxblox::TsdfMap> tsdf_map_;
  boost::shared_ptr<voxblox::FastTsdfIntegrator> tsdf_integrator_;

  /// ICP matcher
  boost::shared_ptr<voxblox::ICP> icp_;
  boost::shared_ptr<voxblox::MeshLayer> mesh_layer_;
  boost::shared_ptr<voxblox::MeshIntegrator<voxblox::TsdfVoxel>>
      mesh_integrator_;

  boost::shared_ptr<utilities::image::VisoFeatureTracker> viso_extractor_;

  boost::shared_ptr<utilities::image::OrbFeatureMatcher> orb_matcher_;

  boost::shared_ptr<utilities::transform::TransformMap> transform_map_;

  cv::Mat curr_descriptor_;

  std::vector<utilities::transform::Transform> pose_chain_;

  utilities::transform::Transform curr_pose_;

  std::vector<Eigen::Vector3d> curr_pointcloud_;
  std::vector<Eigen::Vector3d> color_pointcloud_;

  Eigen::Affine3f current_position_;

  std::vector<Eigen::Affine3f> position_vector_;

  utilities::image::StereoImage curr_frame_;

  boost::shared_ptr<utilities::viewer::Viewer> viewer_;

  bool use_laser_;

  int frame_no;

};


} // namespace mesh_map
} // namespace mapping
} // namespace core
} // namespace aru

#endif // ARU_CORE_MAPPING_MESH_MAP_H_
