
#define SLOW_BUT_CORRECT_BETWEENFACTOR
#pragma once

#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>

//gtsam header files
#include <Eigen/Dense>
#include <Eigen/Core>
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
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/dataset.h>










namespace pgmo{

    //deformation edge taken from https://github.com/MIT-SPARK/Kimera-PGMO/blob/master/include/kimera_pgmo/DeformationGraph.h

    class DeformationEdgeFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>{
        private: 
        gtsam::Pose3 node1_pose;
        gtsam::Point3 node2_position;

        //Define unique factor for the deformation edges
        public:

        DeformationEdgeFactor(gtsam::Key node1_symbol, gtsam::Key node2_symbol, const gtsam::Pose3& node1_pose, const gtsam::Point3& node2_point, gtsam::SharedNoiseModel model) : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model,node1_symbol,node2_symbol),                                                   
        node1_pose(node1_pose), node2_position(node2_point) {}

        ~DeformationEdgeFactor() {}

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

    class DeformationGraph
    {
        private:
        gtsam::NonlinearFactorGraph::shared_ptr graph;
        gtsam::Values::shared_ptr deformationGraph;
        std::ofstream outfile;


        public:
        //constructor
        DeformationGraph(std::string filename, std::string dfg_filename) 
        {
            // Read graph from file
            filename = "../data/"+filename;

            bool is3D = true;
            boost::tie(graph, deformationGraph) = gtsam::readG2o(filename, is3D);

            //open dfg file for writing
            outfile = std::ofstream(dfg_filename,std::ios::out);
        }
        //destructor
        ~DeformationGraph() {}
        //add odometry edge factor
        int AddOdomEdge(gtsam::Key sym1, gtsam::Key sym2, gtsam::Pose3 odometry)
        {
            auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << gtsam::Vector3::Constant(0.01), gtsam::Vector3::Constant(0.03)).finished());
            graph->add(gtsam::BetweenFactor<gtsam::Pose3>(sym1, sym2, odometry, poseNoise));
            return 0;
        }
        //add prior factor
        int AddPrior()
        {
            //prior noise model
            auto priorModel = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            for (const auto keyVal : deformationGraph->keys())
            {
                gtsam::Pose3 anchorPose = deformationGraph->at<gtsam::Pose3>(keyVal);
                graph->addPrior(keyVal, anchorPose, priorModel);
                break;
            }
            return 0;
        }
        //write values to a dfg file
        int writeDFG(gtsam::Values vals)
        {
            if (!outfile)
            {
                std::cout << "error writing to file" << std::endl;
                return 0;
            }

            std::vector<gtsam::Key> keys;
            int counter=0;
            bool mesh_vertices=false;

            for (auto key : vals.keys())
            {
                keys.push_back(key);
                if (key==1661)
                {
                    mesh_vertices=true;
                    outfile << "NODE " << key << " " << vals.at<gtsam::Pose3>(key).translation().x() << " " << vals.at<gtsam::Pose3>(key).translation().y() << " " << vals.at<gtsam::Pose3>(key).translation().z() << std::endl;
                }
                else
                {
                    if (mesh_vertices)
                    {
                        outfile << "NODE " << key << " " << vals.at<gtsam::Pose3>(key).translation().x() << " " << vals.at<gtsam::Pose3>(key).translation().y() << " " << vals.at<gtsam::Pose3>(key).translation().z() << std::endl;
                    }
                    else
                    {
                        if (counter==0)
                        {
                            outfile << "NODE " << key << " " << vals.at<gtsam::Pose3>(key).translation().x() << " " << vals.at<gtsam::Pose3>(key).translation().y() << " " << vals.at<gtsam::Pose3>(key).translation().z() << std::endl;
                        }
                        else
                        {
                            outfile << "NODE " << key << " " << vals.at<gtsam::Pose3>(key).translation().x() << " " << vals.at<gtsam::Pose3>(key).translation().y() << " " << vals.at<gtsam::Pose3>(key).translation().z() << std::endl;
                            outfile << "BETWEEN " << keys[counter] << " " << keys[counter-1] << std::endl;
                        }

                    }
                }

                counter++;
            }

            return 1;

        }

        //close file
        int closeDFG()
        {
            outfile.close();
            return 1;
        }
        //add deformation edge factor between two mesh nodes
        int AddMeshEdge(gtsam::Key v_from, gtsam::Key v_to)
        {
            double variance =1e-02;
            static const gtsam::SharedNoiseModel& noise = gtsam::noiseModel::Isotropic::Variance(3, variance);
            const DeformationEdgeFactor edge1(v_from, v_to, deformationGraph->at<gtsam::Pose3>(v_from),deformationGraph->at<gtsam::Pose3>(v_to).translation(),noise);
            //const DeformationEdgeFactor edge2(v_to, v_from, deformationGraph->at<gtsam::Pose3>(v_to),deformationGraph->at<gtsam::Pose3>(v_from).translation(),noise);
            graph->add(edge1);

            //add edge to dfg file
            outfile << "DEDGE " << v_from << " " << v_to << std::endl;
            //graph->add(edge2);
            return 0;
        }
        //add edge between pose and observed mesh vertices
        int AddMeshPoseEdge(gtsam::Key pose_key, std::vector<gtsam::Key> vertices)
        {
            double variance =1e-02;
            static const gtsam::SharedNoiseModel& noise = gtsam::noiseModel::Isotropic::Variance(3, variance);
            for (auto vertex : vertices)
            {
                const DeformationEdgeFactor edge1(pose_key, vertex, deformationGraph->at<gtsam::Pose3>(pose_key),deformationGraph->at<gtsam::Pose3>(vertex).translation(),noise);
                const DeformationEdgeFactor edge2(vertex, pose_key, deformationGraph->at<gtsam::Pose3>(vertex),deformationGraph->at<gtsam::Pose3>(pose_key).translation(),noise);
                graph->add(edge1);
                graph->add(edge2);

                //add edge to dfg file
                outfile << "DEDGE " << pose_key << " " << vertex << std::endl;
            }

            return 0;
        }

        //create poses in a circular trajectory
        std::vector<gtsam::Pose3> createPoses(const gtsam::Pose3 &init = gtsam::Pose3(gtsam::Rot3::Ypr(M_PI / 2, 0, -M_PI / 2),gtsam::Point3(30, 0, 0)),
        const gtsam::Pose3 &delta = gtsam::Pose3(gtsam::Rot3::Ypr(0, -M_PI / 4, 0), gtsam::Point3(sin(M_PI / 4) * 30, 0, 30 * (1 - sin(M_PI / 4)))),
        int steps = 8)
        {
            std::vector<gtsam::Pose3> poses;
            int i = 1;
            poses.push_back(init);

            for(; i < steps; ++i)
            {
                gtsam::Pose3 newPose = poses[i - 1].compose(delta);

                poses.push_back(newPose);
            }

            return poses;
        }

        //add mesh vertices to toy example from file to deformation graph
        std::map<gtsam::Key,gtsam::Point3> AddMeshVertices()
        {
            std::map<gtsam::Key,gtsam::Point3> vertices;

            gtsam::Key mesh_vertex_indx(1661);
          

            for (auto key : deformationGraph->keys())
            {
                std::vector<gtsam::Key> connected_mesh_vertices;

                gtsam::Pose3 pose = deformationGraph->at<gtsam::Pose3>(key);

                gtsam::Point3 position = pose.translation();

                //mesh vertex offset from pose node
                gtsam::Point3 delta(0,0,5);
                gtsam::Point3 position1;
                position1=position+delta;

                //add mesh vertex position to vector
                vertices.insert(std::pair<gtsam::Key,gtsam::Point3>(mesh_vertex_indx,position1));
                //add mesh vertex to valences
                connected_mesh_vertices.push_back(mesh_vertex_indx);
               
                //add mesh vertex pose 
                gtsam::Pose3 meshPose(gtsam::Rot3(),position1);

                //insert mesh Pose into initial values
                deformationGraph->insert(mesh_vertex_indx, meshPose);

                mesh_vertex_indx++;

                //mesh vertex offset from pose node
                gtsam::Point3 delta1(0,2,5);
                gtsam::Point3 position2;
                position2=position+delta1;

                //add mesh vertex position to vector
                vertices.insert(std::pair<gtsam::Key,gtsam::Point3>(mesh_vertex_indx,position2));
                //add mesh vertex to valences
                connected_mesh_vertices.push_back(mesh_vertex_indx);
                

                //add mesh vertex pose 
                gtsam::Pose3 meshPose1(gtsam::Rot3(),position2);

                //insert mesh Pose into initial values
                deformationGraph->insert(mesh_vertex_indx, meshPose1);

                //add deformation edges between keyframe pose and observed vertices
                int result = AddMeshPoseEdge(key,connected_mesh_vertices);

                //add deformation edges between mesh vertices
                int result1 = AddMeshEdge(connected_mesh_vertices[0], connected_mesh_vertices[1]);

                //increment key
                mesh_vertex_indx++;
                
            }
            return vertices;

        }



        //optimize the graph using Gauss-Newton
        int optimizeGN()
        {
            //optimise the graph using gauss-newton
            gtsam::GaussNewtonParams params;
            // Create the optimizer ...
            gtsam::GaussNewtonOptimizer opt(*graph, *deformationGraph, params);
            gtsam::Values results=opt.optimize();

            return 1;
        }

        //optimize the graph using Levenberg-Marquadt
        gtsam::Values optimizeLM()
        {
            gtsam::LevenbergMarquardtParams params;
            gtsam::LevenbergMarquardtParams::SetCeresDefaults(&params);
            params.setVerbosity("ERROR");
            gtsam::LevenbergMarquardtOptimizer opt(*graph, *deformationGraph, params);
            gtsam::Values results=opt.optimize();
            return results;
        }

        //print out mesh vertices
        int printVertices(std::map<gtsam::Key,gtsam::Point3>& vertices)
        {
            for (auto vertex : vertices)
            {
                std::cout << "Key: " << vertex.first << " " << "[" << vertex.second.x() << "," << vertex.second.y() << "," << vertex.second.z() << "]" << std::endl;
            }
        }


        //return copy of deformation graph values
        gtsam::Values valsCopy()
        {
            gtsam::Values copy(*deformationGraph);
            return copy;
        }



        




    };

  
}