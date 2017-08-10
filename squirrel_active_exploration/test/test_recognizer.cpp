#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <sstream>
#include <algorithm>
#include <tf/transform_listener.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <pcl/filters/passthrough.h>
#include <pcl_ros/transforms.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/passthrough.h>

#include <squirrel_object_perception_msgs/SegmentInit.h>
#include <squirrel_object_perception_msgs/SegmentOnce.h>
#include <squirrel_object_perception_msgs/SegmentsToObjects.h>
#include <squirrel_object_perception_msgs/SegmentVisualizationInit.h>
#include <squirrel_object_perception_msgs/SegmentVisualizationOnce.h>
#include <squirrel_object_perception_msgs/Recognize.h>

#include "squirrel_active_exploration/io_utils.h"

using namespace std;
using namespace pcl;

vector<int> read_indices(string &filename);

int main(int argc, char** argv)
{
    ros::init(argc, argv, "recognize_object");
    ros::NodeHandle n ("~");
    ROS_INFO("%s: started node", ros::this_node::getName().c_str());

    // Set up the service
    string recognizer_topic = "/squirrel_recognize_objects";
    ros::ServiceClient client = n.serviceClient<squirrel_object_perception_msgs::Recognize>(recognizer_topic);
    squirrel_object_perception_msgs::Recognize srv;

    // Load the input point cloud
    string cloud_name = "/home/tpatten/Data/models/TUW_models/TUW_models/asus_box/views/cloud_00000000.pcd";
    PointCloud<PointT> cloud;
    if (io::loadPCDFile<PointT> (cloud_name.c_str(), cloud) == -1)
    {
        ROS_ERROR("Could not load point cloud %s", cloud_name.c_str());
        return EXIT_FAILURE;
    }
    ROS_INFO("Read %lu points", cloud.size());
    cloud.height = 480;
    cloud.width = 640;
    // Load the indices
    string indices_file = "/home/tpatten/Data/models/TUW_models/TUW_models/asus_box/views/object_indices_00000000.txt";
    vector<int> seg = read_indices(indices_file);
    if (seg.size() <= 0)
    {
        ROS_ERROR("Could not read index file");
        return EXIT_FAILURE;
    }
    else
    {
        ROS_INFO("Read %lu indices", seg.size());
    }
    // Crop the point cloud
    PointCloud<PointT> segmented_cloud;
    copyPointCloud(cloud, seg, segmented_cloud);
    //pcl::io::savePCDFileASCII ("/home/tpatten/Data/segmented.pcd", segmented_cloud);
    PointT min_p, max_p;
    getMinMax3D(segmented_cloud, min_p, max_p);
    cout << "Size from Segmenter: " << "X(" << min_p.x << ";" << max_p.x << ")"
         << " Y(" << min_p.y << ";" << max_p.y << ")"
         << " Z(" << min_p.z << ";" << max_p.z << ")" << endl;

    PointCloud<PointT>::Ptr cloud_ptr (new PointCloud<PointT>(cloud));
    PassThrough<PointT> pass;
    pass.setKeepOrganized(true);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(min_p.x-0.05, max_p.x+0.05);
    pass.setInputCloud(cloud_ptr);
    pass.filter(*cloud_ptr);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(min_p.y-0.05, max_p.y+0.05);
    pass.setInputCloud(cloud_ptr);
    pass.filter(*cloud_ptr);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_p.z-0.05, max_p.z+0.05);
    pass.setInputCloud(cloud_ptr);
    pass.filter(*cloud_ptr);
    //pcl::io::savePCDFileASCII ("/home/tpatten/Data/filtered.pcd", *cloud_ptr);

    // Recognition
    toROSMsg(*cloud_ptr, srv.request.cloud);
    cout << srv.request.cloud.height << endl;
    cout << srv.request.cloud.width << endl;
    if (client.call(srv))
    {
        ROS_INFO("Called service %s: ", recognizer_topic.c_str());
        cout << srv.response.confidence.size() << endl;
        if (srv.response.ids.size() > 0)
        {
            for (int i= 0; i < srv.response.ids.size(); i++)
            {
                /*
                # model ids of recognized objects
                std_msgs/String[] ids

                # 3D poses of recognized objects
                geometry_msgs/Transform[] transforms

                # confidence value defined as ratio of visible points
                float32[] confidence

                # centroid of the cluster
                geometry_msgs/Point32[] centroid

                # bounding box of the cluster
                squirrel_object_perception_msgs/BBox[] bbox

                # point cloud of the model transformed into camera coordinates
                sensor_msgs/PointCloud2[] models_cloud
                */
                cout << "-- Object " << i << "/" << srv.response.ids.size() << endl;
                cout << "Category: " << srv.response.ids.at(i).data << endl;
                cout << "Confidence: " << srv.response.confidence.at(i) << endl;
                //object.pose = transform(srv.response.centroid.at(i).x, srv.response.centroid.at(i).y, srv.response.centroid.at(i).z,
                //                        srv.request.cloud.header.frame_id, "/map").pose;
            }
        }
        else
        {
            ROS_WARN("Could not recognize an object!");
        }
    }
    else
    {
        ROS_ERROR("Could not call service");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

vector<int> read_indices(string &filename)
{
    vector<int> indices;

    ifstream myfile (filename.c_str());
    if (myfile.is_open())
    {
        int in;
        while (myfile >> in)
        {
            indices.push_back(in);
        }
        myfile.close();
    }

    return indices;

}
