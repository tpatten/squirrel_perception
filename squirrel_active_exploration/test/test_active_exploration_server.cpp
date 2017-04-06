#include <ros/ros.h>
#include <octomap_ros/conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <squirrel_object_perception_msgs/Segment.h>
#include <squirrel_object_perception_msgs/SegmentVisualizationInit.h>
#include <squirrel_object_perception_msgs/SegmentVisualizationOnce.h>
#include <squirrel_object_perception_msgs/Classify.h>
#include <squirrel_object_perception_msgs/ActiveExplorationNBV.h>
#include "squirrel_active_exploration/io_utils.h"

using namespace std;
using namespace pcl;
using namespace octomap;

// Default definitions, should be replaced by program arguments
#define _VARIANCE 0.5
#define _TREE_DEPTH 14  // 14 => resolution of 0.1m, initial tree depth is 16 => 0.025m
#define _CAMERA_HEIGHT 0.75  // height of camera above robot base
#define _ROBOT_RADIUS 0.22
#define _DISTANCE_FROM_CENTER 2
#define _NUM_LOCATIONS 10
#define _TREE_RESOLUTION 0.1

// Function declarations
bool load_data(const string &data_name, const bool &reverse_transforms, vector<Eigen::Vector4f> &poses,
               vector<PointCloud<PointT> > &clouds, vector<vector<vector<int> > > &indices);

// Run Main
int main(int argc, char **argv)
{
    ros::init (argc, argv ,"test_active_exploration_server");
    ros::NodeHandle n("~");

    // Get the parameters
    string data_dir = "";
    string train_dir = "";
    string saliency_filename = "";
    double variance = _VARIANCE;
    double camera_height = _CAMERA_HEIGHT;
    double robot_radius = _ROBOT_RADIUS;
    double distance_from_center = _DISTANCE_FROM_CENTER;
    int num_locations = _NUM_LOCATIONS;
    double tree_resolution = _TREE_RESOLUTION;
    bool reverse_transforms = false;
    bool occlusions = true;
    bool working_classifier = false;
    bool multiple_locations = false;
    // Read the input if it exists
    if (!n.getParam ("data_dir", data_dir))
    {
        ROS_ERROR("test_active_exploration_server : you must enter a directory or filename!");
        return EXIT_FAILURE;
    }
    n.getParam ("train_dir", train_dir);
    n.getParam ("saliency_map", saliency_filename);
    n.getParam ("variance", variance);
    n.getParam ("camera_height", camera_height);
    n.getParam ("robot_radius", robot_radius);
    n.getParam ("distance_from_center", distance_from_center);
    n.getParam ("num_locations", num_locations);
    n.getParam ("tree_resolution", tree_resolution);
    n.getParam ("reverse_transforms", reverse_transforms);
    n.getParam ("occlusions", occlusions);
    n.getParam ("working_classifier", working_classifier);
    // Print out the input
    ROS_INFO("test_active_exploration_server : input parameters");
    cout << "Data directory = " << data_dir << endl;
    cout << "Variance = " << variance << endl;
    cout << "Camera height = " << camera_height << endl;
    cout << "Robot radius = " << robot_radius << endl;
    cout << "Distance from center = " << distance_from_center << endl;
    cout << "Num locations = " << num_locations << endl;
    cout << "Tree resolution = " << tree_resolution << endl;
    if (reverse_transforms)
        cout << "Reverse transforms = TRUE" << endl;
    else
        cout << "Reverse transforms = FALSE" << endl;
    if (occlusions)
        cout << "Occlusions = TRUE" << endl;
    else
        cout << "Occlusions = FALSE" << endl;
    if (working_classifier)
        cout << "Classifier = TRUE" << endl;
    else
        cout << "Classifier = FALSE" << endl;
    // If classifier is not working, then needs a valid training directory
    if (!working_classifier)
    {
        if (train_dir.size() == 0)
        {
            ROS_ERROR("test_active_exploration_server : classifier is assumed not working, you must enter a training directory");
            return EXIT_FAILURE;
        }
        // Add backslash
        train_dir = add_backslash(train_dir);
    }

    // Load the clouds from the file
    vector<Eigen::Vector4f> poses;
    vector<PointCloud<PointT> > clouds;
    vector<vector<vector<int> > > indices;
    if (!load_data(data_dir, reverse_transforms, poses, clouds, indices))
    {
        ROS_ERROR("test_active_exploration_server : could not load the data");
        return EXIT_FAILURE;
    }
    // Get a single pose, cloud and transform
    Eigen::Vector4f pose;
    PointCloud<PointT> cloud;
    vector<vector<int> > segs;
    if (clouds.size() == 0)
    {
        ROS_ERROR("test_active_exploration_server : clouds is empty");
        return EXIT_FAILURE;
    }
    else if (clouds.size() == 1)
    {
        ROS_INFO("test_active_exploration_server : loaded one cloud");
        pose = poses[0];
        cloud = clouds[0];
        if (indices.size() > 0)
            segs = indices[0];
    }
    else
    {
        ROS_INFO("test_active_exploration_server : loaded multiple clouds");
        multiple_locations = true;
        int r = int_rand(0, clouds.size()-1);
        if (r < 0 || r >= clouds.size())
        {
            ROS_ERROR("test_active_exploration_server : invalid random number %i for poses of size %lu", r, poses.size());
            return EXIT_FAILURE;
        }
        // Otherwise get the elements
        pose = poses[r];
        cloud = clouds[r];
        if (r < indices.size())
            segs = indices[r];
    }

    // Create the service clients
    ROS_INFO("test_active_exploration_server : setting up service clients");
    ros::ServiceClient nbv_client = n.serviceClient<squirrel_object_perception_msgs::ActiveExplorationNBV>("/squirrel_active_exploration");
    squirrel_object_perception_msgs::ActiveExplorationNBV nbv_srv;
    ros::ServiceClient seg_client = n.serviceClient<squirrel_object_perception_msgs::Segment>("/squirrel_segmentation");;
    squirrel_object_perception_msgs::Segment seg_srv;
    ros::ServiceClient classify_client = n.serviceClient<squirrel_object_perception_msgs::Classify>("/squirrel_esf_classify");
    squirrel_object_perception_msgs::Classify classify_srv;
    ros::ServiceClient seg_vis_init_client = n.serviceClient<squirrel_object_perception_msgs::SegmentVisualizationInit>("/squirrel_segmentation_visualization_init");
    squirrel_object_perception_msgs::SegmentVisualizationInit seg_vis_init_srv;
    ros::ServiceClient seg_vis_client = n.serviceClient<squirrel_object_perception_msgs::SegmentVisualizationOnce>("/squirrel_segmentation_visualization_once");
    squirrel_object_perception_msgs::SegmentVisualizationOnce seg_vis_srv;

    // Convert the point cloud to a ros message type
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);

    // Initialize the the segmentation viewer
    cv::Mat saliency = cv::imread(saliency_filename,-1);
    cv_bridge::CvImagePtr cv_ptr (new cv_bridge::CvImage);
    ros::Time time = ros::Time::now();
    // Convert OpenCV image to ROS message
    cv_ptr->header.stamp = time;
    cv_ptr->header.frame_id = "saliency_map";
    cv_ptr->encoding = "mono8";
    cv_ptr->image = saliency;
    sensor_msgs::Image saliency_map;
    cv_ptr->toImageMsg(saliency_map);
    seg_vis_init_srv.request.cloud = cloud_msg;
    seg_vis_init_srv.request.saliency_map = saliency_map;
    if (!seg_vis_init_client.call(seg_vis_init_srv))
    {
        ROS_ERROR("test_active_exploration_server : failed to call service /squirrel_segmentation_visualization_init");
        return EXIT_FAILURE;
    }

    // Create an octree with the point cloud input
    OcTree tree (tree_resolution);
    // Add the point cloud
    // Otherwise insert the point cloud
    point3d pos (pose[0], pose[1], pose[2]);
    octomap::Pointcloud o_cloud;
    pointCloud2ToOctomap(cloud_msg, o_cloud);
    tree.insertPointCloud(o_cloud, pos);

    // Octomap
    octomap_msgs::Octomap oc_msg;
    if (!octomap_msgs::binaryMapToMsg(tree, oc_msg))
    {
        ROS_ERROR("test_active_exploration_server : could not convert the octomap");
        return EXIT_FAILURE;
    }

    // Segmentation
    vector<std_msgs::Int32MultiArray> clusters_indices;
    if (segs.size() > 0)
    {
        // From preloaded data
        ROS_INFO("test_active_exploration_server : using preloaded segmentation data");
        for (size_t i = 0; i < segs.size(); ++i)
        {
            std_msgs::Int32MultiArray s;
            s.data = segs[i];
            clusters_indices.push_back(s);
        }
    }
    else
    {
        // Call the segmentation service
        ROS_INFO("test_active_exploration_server : segmenting point cloud with %lu points", cloud_msg.data.size());
        seg_srv.request.cloud = cloud_msg;
        if (!seg_client.call(seg_srv))
        {
            ROS_ERROR("test_active_exploration_server : could not call the segmentation service");
            return EXIT_FAILURE;
        }
        clusters_indices = seg_srv.response.clusters_indices;

        for (size_t i = 0; i < seg_srv.response.clusters_indices.size(); ++i)
        {
            cout << "Segment " << i << " - " << seg_srv.response.clusters_indices[i].data.size() << endl;
            // Visualise
            seg_vis_srv.request.clusters_indices.resize(1);
            seg_vis_srv.request.clusters_indices[0] = seg_srv.response.clusters_indices[i];
            if (!seg_vis_client.call(seg_vis_srv))
            {
                ROS_ERROR("test_active_exploration_server : failed to call service /squirrel_segmentation_visualization_once");
                return EXIT_FAILURE;
            }
        }
    }

    // If segmentation failed, try a random patch of points
    if (clusters_indices.size() == 0)
    {
        ROS_WARN("test_active_exploration_server : failed to do proper segmentation, proceeding with a random group of points from centre");
        PointT min, max;
        getMinMax3D(cloud, min, max);
        PointT search_point;
        search_point.x = (max.data[0] - min.data[0])/2;
        search_point.y = (max.data[1] - min.data[1])/2;
        search_point.z = (max.data[2] - min.data[2])/2;
        KdTreeFLANN<PointT> kdtree;
        PointCloud<PointT>::Ptr cloud_ptr (new PointCloud<PointT>(cloud));
        kdtree.setInputCloud (cloud_ptr);
        vector<int> point_idx;
        vector<float> point_squared_distances;
        float radius = 50; // cm??
        if (kdtree.radiusSearch (search_point, radius, point_idx, point_squared_distances) > 0)
        {
            ROS_WARN("test_active_exploration_server : segment has %lu points", point_idx.size());
            std_msgs::Int32MultiArray s;
            s.data = point_idx;
            clusters_indices.push_back(s);
        }
    }

    // Classify
    vector<squirrel_object_perception_msgs::Classification> class_results;
    if (working_classifier)
    {
        ROS_INFO("test_active_exploration_server_robot : calling classifier");
        classify_srv.request.cloud = cloud_msg;
        classify_srv.request.clusters_indices = clusters_indices;
        if (!classify_client.call(classify_srv))
        {
            ROS_ERROR("test_active_exploration_server : could not call the classification service");
            return EXIT_FAILURE;
        }
        class_results = classify_srv.response.class_results;
        cout << "Returned " << class_results.size() << " results" << endl;
        for (size_t i = 0; i < class_results.size(); ++i)
        {
            ROS_INFO("Segment %lu -", i);
            for (size_t j = 0; j < class_results[i].class_type.size(); ++j)
                //ROS_INFO("  %-15s %s %.2f", class_results[i].class_type[j].data.c_str(), class_results[i].pose[j].data.c_str(), class_results[i].confidence[j]);
                ROS_INFO("  %-15s %.2f", class_results[i].class_type[j].data.c_str(), class_results[i].confidence[j]);
        }
    }
    else
    {
        // Fake classification if it is not working
        ROS_WARN("test_active_exploration_server_robot : fake classification!");
        for (size_t i = 0; i < clusters_indices.size(); ++i)
        {
            squirrel_object_perception_msgs::Classification c;
            std_msgs::String str;
            str.data = "apple/";
            c.class_type.push_back(str);
            str.data = "bottle/";
            c.class_type.push_back(str);
            str.data = "mug/";
            c.class_type.push_back(str);
            c.confidence.push_back(0.2);
            c.confidence.push_back(0.3);
            c.confidence.push_back(0.5);
            str.data = train_dir + "apple//3a92a256ad1e060ec048697b91f69d2/views/pose_0.txt";
            c.pose.push_back(str);
            str.data = train_dir + "bottle//1cf98e5b6fff5471c8724d5673a063a6/views/pose_0.txt";
            c.pose.push_back(str);
            str.data = train_dir + "mug//1c9f9e25c654cbca3c71bf3f4dd78475/views/pose_0.txt";
            c.pose.push_back(str);
            class_results.push_back(c);
        }
    }

    // Set the fields in the service
    nbv_srv.request.robot_pose.position.x = pose[0];
    nbv_srv.request.robot_pose.position.y = pose[1];
    nbv_srv.request.robot_pose.position.z = pose[2];
    nbv_srv.request.variance = variance;
    nbv_srv.request.cloud = cloud_msg;
    nbv_srv.request.map = oc_msg;
    nbv_srv.request.occlusions = occlusions;
    nbv_srv.request.clusters_indices = clusters_indices;
    nbv_srv.request.class_results = class_results;


    // --- Test 1: given locations
    if (multiple_locations)
    {
        vector<geometry_msgs::Point> locations;
        //for (size_t i = 0; i < poses.size(); ++i)
        for (size_t i = 0; i < 4; ++i)
        {
            geometry_msgs::Point p;
            p.x = poses[i][0];
            p.y = poses[i][1];
            p.z = poses[i][2];
            locations.push_back(p);
        }
        nbv_srv.request.locations = locations;
        // Call the service
        ROS_INFO("test_active_exploration_server : calling next best view service with given locations");
        if (!nbv_client.call(nbv_srv))
        {
            ROS_ERROR("test_active_exploration_server : could not call the next best view service");
            return EXIT_FAILURE;
        }
        // Print out the best index
        cout << endl;
        ROS_INFO("Next best view index is %i", nbv_srv.response.nbv_ix);
        // Print out the locations and their utilities
        cout << "locations and utilities:" << endl;
        for (size_t i = 0; i < nbv_srv.response.generated_locations.size(); ++i)
        {
            cout << "[" << nbv_srv.response.generated_locations[i].x << " "
                 << nbv_srv.response.generated_locations[i].y << " "
                 << nbv_srv.response.generated_locations[i].z << "] -> "
                 << nbv_srv.response.utilities[i] << endl;
        }
    }

    // --- Test 2: without locations (they must be generated)
    nbv_srv.request.locations.clear();
    nbv_srv.request.camera_height = camera_height;
    nbv_srv.request.robot_radius = robot_radius;
    nbv_srv.request.distance_from_center = distance_from_center;
    nbv_srv.request.num_locations = num_locations;
    ROS_INFO("test_active_exploration_server : calling next best view service without given locations");
    if (!nbv_client.call(nbv_srv))
    {
        ROS_ERROR("test_active_exploration_server : could not call the next best view service");
        return EXIT_FAILURE;
    }
    // Print out the best index
    cout << endl;
    ROS_INFO("Next best view index is %i", nbv_srv.response.nbv_ix);
    // Print out the locations and their utilities
    cout << "locations and utilities:" << endl;
    for (size_t i = 0; i < nbv_srv.response.generated_locations.size(); ++i)
    {
        cout << "[" << nbv_srv.response.generated_locations[i].x << " "
             << nbv_srv.response.generated_locations[i].y << " "
             << nbv_srv.response.generated_locations[i].z << "] -> "
             << nbv_srv.response.utilities[i] << endl;
    }

    // End
    ros::shutdown();
    return EXIT_SUCCESS;
}

bool load_data(const string &data_name, const bool &reverse_transforms, vector<Eigen::Vector4f> &poses,
               vector<PointCloud<PointT> > &clouds, vector<vector<vector<int> > > &indices)
{
    // Clear the vectors
    poses.clear();
    clouds.clear();
    indices.clear();

    // If this is a single file, then load it
    if (data_name.size() > 4 && data_name.substr(data_name.size()-4) == ".pcd")
    {
        ROS_INFO("test_active_exploration_server::load_data : single file %s", data_name.c_str());
        // Single point cloud input
        PointCloud<PointT> cloud;
        if (io::loadPCDFile<PointT> (data_name.c_str(), cloud) == -1)
        {
            ROS_ERROR("test_active_exploration_server::load_data : could not load point cloud %s", data_name.c_str());
            return false;
        }
        // Get the directory from the filename
        string cloud_path_name;
        string cloud_filename;
        if (!split_filename(data_name, cloud_path_name, cloud_filename))
        {
            ROS_ERROR("test_active_exploration_server::load_data : could not get path and filename from string %s", data_name.c_str());
            return false;
        }
        // Check if pose and transform are valid
        Eigen::Quaternionf cloud_transform = cloud.sensor_orientation_;
        bool do_transform = false;
        cout << data_name << endl;
        //cout << cloud_transform << endl;
        Eigen::Vector4f pose;
        PointCloud<PointT> transformed_cloud;
        Eigen::Matrix4f transform;
        if (isIdentity(cloud_transform))
        {
            ROS_WARN("test_active_exploration_server::load_data : cloud has an identity transform, searching for transform file");
            do_transform = true;
            // Get the transform
            if (!transform_cloud_from_file(cloud_path_name, cloud_filename, cloud, transformed_cloud, pose, transform))
            {
                ROS_WARN("test_active_exploration_server::load_data : error transforming point cloud from file %s", data_name.c_str());
                return false;
            }
        }

        if (do_transform)
        {
            // Reverse transform
            Eigen::Matrix4f transform_inv = transform;
            if (reverse_transforms)
                transform_inv = transform.inverse();
            transform = transform_inv;
            // Transform the point cloud
            transformPointCloud(cloud, cloud, transform);
            // Get the pose from the transformed point cloud
            pose = extract_camera_position (cloud);
            pose = transform_eigvec(pose, transform);

            // Add to vectors
            poses.push_back(pose);
            clouds.push_back(cloud);
        }
        else
        {
            // Add to vectors
            pose = cloud.sensor_origin_;
            poses.push_back(pose);
            clouds.push_back(cloud);
        }

        // Get the segment indices
        // First get the index number of the point cloud
        // Get the dot
        size_t dot = data_name.find_last_of('.');
        if (dot == string::npos)
        {
            ROS_ERROR("test_active_exploration_server::load_data : could not read extension of file %s", cloud_filename.c_str());
            return false;
        }
        // Get the index name
        int ix = -1;
        size_t underscore = cloud_filename.find_last_of('_');
        if (underscore != string::npos && dot != string::npos && (dot - underscore) > 0)
        {
            int str_len = dot - underscore;
            string str_ix = cloud_filename.substr(underscore+1,str_len-1);
            ix = atoi(str_ix.c_str());
        }
        else
        {
            ROS_ERROR("test_active_exploration_server::load_data : could not read the index in file %s", cloud_filename.c_str());
            return false;
        }
        // Append zeros to front
        string ix_str = boost::lexical_cast<string>(ix);
        while (ix_str.size() < 10)
            ix_str = "0" + ix_str;
        string segment_indices_str;
        vector<vector<int> > segment_indices;
        int seg_count = 0;
        // Load the segment indices for each segment associated to this cloud
        while (true)
        {
            // Create the count string
            string count_str = boost::lexical_cast<string>(seg_count);
            while (count_str.size() < 2)
                count_str = "0" + count_str;
            segment_indices_str = add_backslash(cloud_path_name) + _INDICES_PREFIX + count_str + "_" + ix_str + ".pcd";
            // If valid file then load it
            if (boost::filesystem::exists(segment_indices_str))
            {
                PointCloud<IndexPoint> in_cloud;
                if (io::loadPCDFile<IndexPoint>(segment_indices_str.c_str(), in_cloud) == -1)
                {
                    ROS_WARN("test_active_exploration_server::load_data : could not read index file");
                    break;
                }
                else
                {
                    // Append the point cloud indices
                    vector<int> in_indices;
                    in_indices.resize(in_cloud.points.size());
                    for (size_t j = 0; j < in_cloud.points.size(); ++j)
                        in_indices[j] = in_cloud.points[j].idx;
                    segment_indices.push_back(in_indices);
                }
            }
            // Otherwise finish
            else
            {
                break;
            }
            // Next segment
            ++seg_count;
        }
        indices.push_back(segment_indices);
    }
    // Otherwise it is a directory and must load all files in the directory
    else
    {
//        if (!load_test_directory_with_segment_indices(data_name, reverse_transforms, poses, clouds, transforms, indices))
//        {
//            ROS_ERROR("test_active_exploration_server::load_data : could not load the data from the directory %s", data_name.c_str());
//            return false;
//        }
        // This can also work without getting the segment indices
        if (!load_test_directory(data_name, reverse_transforms, poses, clouds))
        {
            ROS_ERROR("test_active_exploration_server::load_data : could not load the data from the directory %s", data_name.c_str());
            return false;
        }
    }

    // Return success
    return true;
}
