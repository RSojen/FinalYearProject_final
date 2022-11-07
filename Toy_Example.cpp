

#include "DeformationGraph.h"


using namespace gtsam;
using namespace pgmo;
using namespace std;
int main()
{

    //noise for deformation factors
    double variance =1e-02;
    static const SharedNoiseModel& noise =
    noiseModel::Isotropic::Variance(3, variance);

    //noise for 2d example
    auto odomNoise = noiseModel::Diagonal::Sigmas(Vector3(0.5,0.5,0.1));
    auto priorNoise = noiseModel::Diagonal::Sigmas(Vector3(1.0,1.0,0.1));

    //noise model for poses
    auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.03))
          .finished());

    
    //construct deformation graph object
    DeformationGraph graph("parking-garage.g2o","../plots/optimised.dgrf");

    //vector to store initials mesh vertex positions
    std::map<gtsam::Key,gtsam::Point3> mesh_vertices;
    //extract poses
    Values poses=graph.valsCopy();
    //print out poses
    std::map<gtsam::Key,gtsam::Point3> pose_points;
    for (auto pose : poses.keys())
    {
        pose_points.insert(pair<Key, Point3>(pose, poses.at<Pose3>(pose).translation()));
    }
    cout << "printing initial poses..." << endl;
    graph.printVertices(pose_points);
    cout << "finished printing initial poses" << endl;

    //add mesh vertices to deformation graph and print out vertices
    mesh_vertices=graph.AddMeshVertices();
    cout << "printing initial mesh vertices..." << endl;
    graph.printVertices(mesh_vertices);
    cout << "finished printing initial mesh vertices" << endl;
    //extract copy of initials
    Values initials_copy=graph.valsCopy();

    //write to dfg file
    //graph.writeDFG(initials_copy);

    //add a prior
    graph.AddPrior();
    //optimize the graph
    Values finals=graph.optimizeLM();

    std::map<gtsam::Key,gtsam::Point3> optimized_vals;
    for (auto val : finals.keys())
    {
        optimized_vals.insert(pair<Key, Point3>(val, finals.at<Pose3>(val).translation()));
    }
    cout << "printing final values..." << endl;
    graph.printVertices(optimized_vals);
    cout << "finished printing final values" << endl;

    //write to dfg file
    graph.writeDFG(finals);








   
    
    
    
    
    
   

    //deform the mesh using optimization results
    //iterate through deformation graph values 

    vector<multimap<double,Key>> distances;
    

    //store distances to each deformation graph vertex from each mesh vertex
    for (auto v : mesh_vertices)
    {

        multimap<double,Key> node_dist;
        for (auto key : initials_copy.keys())
        {
            //calculate euclidian distance between all mesh vertices
            //deformation graph node
            Point3 node = initials_copy.at<Pose3>(key).translation();
            //mesh vertex position
            Point3 vertex = v.second;
            Point3 dist_vect = node-vertex;
            double dist = dist_vect.norm();
            node_dist.insert(pair<double,Key>(dist,key));
        }
        distances.push_back(node_dist);
    }

    int i = 0;
    int k = 2;
    int counter=0;
    int counter2=0;

    double d_max;

    //vector to store new mesh vertices
    std::map<gtsam::Key,gtsam::Point3> new_vertices;
    
    for (auto v : mesh_vertices)
    {
        Point3 v_new(0.0,0.0,0.0);

      
        
        //extract dmax
        for (auto mapval : distances[i])
        {
            
            if(counter==(k-1))
            {
                d_max=mapval.first;
            }
            counter++;
        }
       
       
        vector<double> weights;

        for (auto mapval: distances[i])
        {

            if (counter2<k)
            {
                //deformation graph node position
                Point3 def_graph_node=initials_copy.at<Pose3>(mapval.second).translation();
                //mesh vertex
                Point3 vertex=v.second;

                Point3 distance_vect=vertex-def_graph_node;
                double dist=distance_vect.norm();

                double weight = (1-dist/d_max);
                weight = weight*weight;
                weights.push_back(weight);
            }

            counter2++;
        }

        //normalize weights
        double max_weight = *max_element(weights.begin(), weights.end());
        for (int i = 0; i < weights.size(); ++i)
        {
            weights[i]=weights[i]/max_weight;
        }


        //deform mesh vertex
        counter2=0;

        for (auto mapval : distances[i])
        {
            if (counter2<k)
            {
                //deformation graph node position
                Point3 def_graph_node=initials_copy.at<Pose3>(mapval.second).translation();
                //mesh vertex
                Point3 vertex=v.second;

                v_new=v_new+weights[counter2]*(finals.at<Pose3>(mapval.second).rotation().rotate(vertex-def_graph_node)+finals.at<Pose3>(mapval.second).translation());
            }
            counter2++;
        }

        //add new vertex to vector
        new_vertices.insert(pair<Key,Point3>(v.first,v_new));
        i++;
        counter2=0;
        counter=0;

    }

    graph.printVertices(new_vertices);
        

}