#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32MultiArray.h>
#include <squirrel_object_perception_msgs/Classify.h>
#include <squirrel_object_perception_msgs/Classification.h>

#include "squirrel_active_exploration/esf_nn_classifier.h"

using namespace std;
using namespace pcl;

class ClassifierROS
{
public:
    ClassifierROS()
    {
        _n = new ros::NodeHandle("~");
        _classify_service = _n->advertiseService("/squirrel_esf_classify", &ClassifierROS::classify, this);

        // Get the model directory and NN
        string models_dir;
        if (!_n->getParam("/squirrel_esf_classifier/models_dir", models_dir))
        {
            ROS_ERROR("ClassifierRos::ClassifierRos : you must set the models directory");
            exit(1);
        }
        int nn;
        _n->param("/squirrel_esf_classifier/nn", nn, 5);
        ROS_INFO("ClassifierRos::ClassifierRos : models_dir %s, nn %i", models_dir.c_str(), nn);

        // Initialize the esf classifier
        esfc.set_model_directory(models_dir);
        esfc.set_NN(nn);
        if (!esfc.initialize())
        {
            ROS_ERROR("ClassifierRos::ClassifierRos : could not initialise esf classifier");
            exit(1);
        }

        ROS_INFO("%s: Ready to receive service calls.", ros::this_node::getName().c_str());
    }

    bool classify(squirrel_object_perception_msgs::Classify::Request &req,
                  squirrel_object_perception_msgs::Classify::Response &response)
    {
      /*
        # the original full view organized point cloud
        sensor_msgs/PointCloud2 cloud

        # the indices of the segmented clusters
        std_msgs/Int32MultiArray[] clusters_indices

        ---

        # classification result for clusters, array corresponds to clusters array
        squirrel_object_perception_msgs/Classification[] class_results

        (
        std_msgs/String[] class_type
        float32[] confidence
        std_msgs/String[] pose
        )

       */

        // Extract the point clouds for each cluster
        PointCloud<PointT> cloud;
        PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(req.cloud, pcl_pc2);
        fromPCLPointCloud2(pcl_pc2, cloud);
        PointCloud<pcl::PointXYZRGB> cloud_rgb;
        copyPointCloud(cloud, cloud_rgb);

        // Set the cloud in the classifier
        esfc.set_cloud(cloud);

        // To return
        vector<squirrel_object_perception_msgs::Classification> class_results;

        // Classify each cluster
        for(size_t cluster_id = 0; cluster_id < req.clusters_indices.size(); cluster_id++)
        {
            const float r = rand() % 255;
            const float g = rand() % 255;
            const float b = rand() % 255;
            vector<int> cluster_indices (req.clusters_indices[cluster_id].data.size());
            for(size_t kk = 0; kk < req.clusters_indices[cluster_id].data.size(); kk++)
            {
                cluster_indices[kk] = static_cast<int>(req.clusters_indices[cluster_id].data[kk]);
                cloud_rgb.at(req.clusters_indices[cluster_id].data[kk]).r = 0.8*r;
                cloud_rgb.at(req.clusters_indices[cluster_id].data[kk]).g = 0.8*g;
                cloud_rgb.at(req.clusters_indices[cluster_id].data[kk]).b = 0.8*b;
            }

            // Classify
            vector<string> class_names;
            vector<float> confidences;
            vector<string> poses;
            esfc.classify(class_names, confidences, poses, cluster_indices);

            // Convert to squirrel object perception message
            squirrel_object_perception_msgs::Classification res;
            std_msgs::String str_tmp;
            cout << "For cluster " << cluster_id << " I have the following hypotheses:" << endl;
            for (size_t i = 0; i < class_names.size(); ++i)
            {
                // Add if confidence > 0
                if (confidences[i] > 0.0 && poses[i] != _NULL_STR)
                {
                    // Class name
                    str_tmp.data = class_names[i];
                    res.class_type.push_back(str_tmp);
                    // Confidence
                    res.confidence.push_back(confidences[i]);
                    // Pose
                    str_tmp.data = poses[i];
                    res.pose.push_back(str_tmp);
                    // Print
                    cout << class_names[i] << " " << confidences[i] << " " << poses[i] << endl;
                }
            }

            // Accumulate result
            class_results.push_back(res);
        }

        // Set the response
        response.class_results = class_results;

        // Return success
        return true;
    }

private:
    ros::NodeHandle *_n;//
    ros::ServiceServer _classify_service;
    ESFNNClassifier esfc;
};

int main (int argc, char ** argv)
{
    ros::init (argc, argv, "squirrel_esf_classifier");
    ClassifierROS m;
    ros::spin();

    return EXIT_SUCCESS;
}
