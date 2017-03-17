#include <ros/ros.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include "squirrel_active_exploration/pcl_conversions.h"
#include <squirrel_object_perception_msgs/Classify.h>

#include "squirrel_active_exploration/active_exploration_utils.h"
#include "squirrel_active_exploration/math_utils.h"

using namespace std;
using namespace pcl;

int main(int argc, char **argv)
{
    ros::init (argc, argv ,"test_classifier");
    ros::NodeHandle n("~");

    // Set up service client for classification
    ros::ServiceClient class_client = n.serviceClient<squirrel_object_perception_msgs::Classify>("/squirrel_classify");
    squirrel_object_perception_msgs::Classify class_srv;

    // Load the input point cloud
    string cloud_name = "/home/tpatten/Data/4DoorSedan.pcd";
    PointCloud<PointT> cloud;
    if (io::loadPCDFile<PointT> (cloud_name.c_str(), cloud) == -1)
    {
        ROS_ERROR("Could not load point cloud %s", cloud_name.c_str());
        return 0;
    }

    // Pass the data to the message
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);

    // Assume all points belong to object (one segment)
    vector<std_msgs::Int32MultiArray> seg_msg;
    seg_msg.resize(1);
    vector<vector<int> > segments;
    segments.resize(1);
    for (size_t i = 0; i < cloud.size(); ++i)
        segments[0].push_back(i);
    for(vector<vector<int> >::size_type i = 0; i < segments.size(); ++i)
    {
        std_msgs::Int32MultiArray seg;
        seg.data = segments[i];
        seg_msg[i] = seg;
    }
    class_srv.request.cloud = cloud_msg;
    class_srv.request.clusters_indices = seg_msg;

    // Call the service
    if (!class_client.call(class_srv))
    {
        ROS_ERROR("Could not call the classification service");
        return 1;
    }
    ROS_INFO("Successfully classified the segments");
    vector<squirrel_object_perception_msgs::Classification> class_estimates = class_srv.response.class_results;

    // Print results
    ROS_INFO("Class estimates");
    for (size_t i = 0; i < class_estimates.size(); ++i)
    {
        ROS_INFO("Segment %lu -", i);
        for (size_t j = 0; j < class_estimates[i].class_type.size(); ++j)
            ROS_INFO("  %-15s %.2f", class_estimates[i].class_type[j].data.c_str(), class_estimates[i].confidence[j]);
    }

    // Finished
    return 0;
}
