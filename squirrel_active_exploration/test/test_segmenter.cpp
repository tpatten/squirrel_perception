#include <ros/ros.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "squirrel_active_exploration/pcl_conversions.h"
#include <squirrel_object_perception_msgs/Segment.h>
#include <squirrel_object_perception_msgs/SegmentInit.h>
#include <squirrel_object_perception_msgs/SegmentOnce.h>
#include <squirrel_object_perception_msgs/SegmentsToObjects.h>
#include <squirrel_object_perception_msgs/SegmentVisualizationInit.h>
#include <squirrel_object_perception_msgs/SegmentVisualizationOnce.h>

using namespace std;
using namespace pcl;

typedef pcl::PointXYZRGB PointT;

int main(int argc, char **argv)
{
    ros::init (argc, argv ,"test_segmenter");
    ros::NodeHandle n("~");

    // Set up service clients for segmentation and visualisation
    ros::ServiceClient seg_init_client = n.serviceClient<squirrel_object_perception_msgs::SegmentInit>("/squirrel_segmentation_incremental_init");
    //ros::ServiceClient seg_client = n.serviceClient<squirrel_object_perception_msgs::SegmentOnce>("/squirrel_segmentation_incremental_once");
    ros::ServiceClient seg_vis_init_client = n.serviceClient<squirrel_object_perception_msgs::SegmentVisualizationInit>("/squirrel_segmentation_visualization_init");
    ros::ServiceClient seg_vis_client = n.serviceClient<squirrel_object_perception_msgs::SegmentVisualizationOnce>("/squirrel_segmentation_visualization_once");
    squirrel_object_perception_msgs::SegmentInit seg_init_srv;
    //squirrel_object_perception_msgs::SegmentOnce seg_srv;
    squirrel_object_perception_msgs::SegmentVisualizationInit seg_vis_init_srv;
    squirrel_object_perception_msgs::SegmentVisualizationOnce seg_vis_srv;

    ros::ServiceClient seg_client = n.serviceClient<squirrel_object_perception_msgs::Segment>("/squirrel_segmentation");
    squirrel_object_perception_msgs::Segment seg_srv;

    // Read the saliency map
    string saliency_name = "/home/tpatten/catkin_ws/squirrel/src/squirrel_perception/squirrel_segmentation/data/test45.png";
    cv::Mat saliency = cv::imread(saliency_name,-1);
    cv_bridge::CvImagePtr cv_ptr (new cv_bridge::CvImage);
    ros::Time time = ros::Time::now();
    // Convert OpenCV image to ROS message
    cv_ptr->header.stamp = time;
    cv_ptr->header.frame_id = "saliency_map";
    cv_ptr->encoding = "mono8";
    cv_ptr->image = saliency;
    sensor_msgs::Image saliency_map;
    cv_ptr->toImageMsg(saliency_map);

    // Load the input point cloud
    string cloud_name;
    if (!n.getParam("cloud", cloud_name))
    {
        ROS_ERROR("You need to specify a point cloud file!");
        return EXIT_FAILURE;
    }
    PointCloud<PointT> cloud;
    if (io::loadPCDFile<PointT> (cloud_name.c_str(), cloud) == -1)
    {
        ROS_ERROR("Could not load point cloud %s", cloud_name.c_str());
        return EXIT_FAILURE;
    }

    // Pass the data to the message
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);

    // Initialise
//    seg_init_srv.request.cloud = cloud_msg;
//    if (!seg_init_client.call(seg_init_srv))
//    {
//        ROS_ERROR("Failed to call service /squirrel_segmentation_incremental_init");
//        return EXIT_FAILURE;
//    }

    seg_vis_init_srv.request.cloud = cloud_msg;
    seg_vis_init_srv.request.saliency_map = saliency_map;
    if (!seg_vis_init_client.call(seg_vis_init_srv))
    {
        ROS_ERROR("Failed to call service /squirrel_segmentation_visualization_init");
        return EXIT_FAILURE;
    }

//    // Extract multiple segments
//    for (int i = 0; i < 6; ++i)
//    {
//        // Get segment
//        if (!seg_client.call(seg_srv))
//        {
//            ROS_ERROR("Failed to call service /squirrel_segmentation_incremental_once");
//            return EXIT_FAILURE;
//        }

//        // Visualise
//        seg_vis_srv.request.clusters_indices = seg_srv.response.clusters_indices;
//        if (!seg_vis_client.call(seg_vis_srv))
//        {
//            ROS_INFO("Failed to call service /squirrel_segmentation_visualization_once");
//            return EXIT_FAILURE;
//        }
//        cout << "Segment " << i << " - " << seg_srv.response.clusters_indices.size() << endl;
//    }

    // Get segments
    seg_srv.request.cloud = cloud_msg;
    if (!seg_client.call(seg_srv))
    {
        ROS_ERROR("Failed to call service /squirrel_segmentation");
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < seg_srv.response.clusters_indices.size(); ++i)
    {
        cout << "Segment " << i << " - " << seg_srv.response.clusters_indices[i].data.size() << endl;
        // Visualise
        seg_vis_srv.request.clusters_indices.resize(1);
        seg_vis_srv.request.clusters_indices[0] = seg_srv.response.clusters_indices[i];
        if (!seg_vis_client.call(seg_vis_srv))
        {
            ROS_ERROR("Failed to call service /squirrel_segmentation_visualization_once");
            return EXIT_FAILURE;
        }
        cin.ignore();
    }

    // Finished
    return EXIT_SUCCESS;
}
