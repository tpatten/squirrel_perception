/*
 * squirrel_segmentation.cpp
 *
 *  Created on: Nov 7, 2014
 *      Author: Ekaterina Potapova
 */

#include <squirrel_segmentation/squirrel_segmentation.hpp>
  
bool
SegmenterComplete::segment (squirrel_object_perception_msgs::Segment::Request &req, squirrel_object_perception_msgs::Segment::Response &response)
{
  //get point cloud
	pcl::PointCloud<PointT>::Ptr inCloud(new pcl::PointCloud<PointT>());
  pcl::fromROSMsg (req.cloud, *inCloud);
	ROS_INFO ("Number of points in the scene: %ld", inCloud->points.size());
  std::vector<pcl::PointIndices> cluster_indices;
  std::vector<pcl::PointCloud<PointT>::Ptr> clusters;

  segmenter_->setInputCloud(inCloud);
  segmenter_->segment();
  segmenter_->getSegmentIndices(cluster_indices);
    
  ROS_INFO ("Number of segmented objects: %ld", cluster_indices.size());
  
	response.clusters_indices.clear();
  for(size_t i = 0; i < cluster_indices.size(); i++)
  {
		  std_msgs::Int32MultiArray indx;
			std::cout << "segment " << i << " has " << cluster_indices[i].indices.size() << " points" << std::endl;
		  for(size_t k = 0; k < cluster_indices[i].indices.size(); k++)
		    	indx.data.push_back(cluster_indices[i].indices[k]);

			// Add to output
		  response.clusters_indices.push_back(indx);
  }
    
  return true;
}

SegmenterComplete::SegmenterComplete ()
{
  //default values
}

SegmenterComplete::~SegmenterComplete ()
{
  if(n_)
    delete n_;
}

void
SegmenterComplete::initialize (int argc, char ** argv)
{
  ros::init (argc, argv, "squirrel_segmentation_server");
  n_ = new ros::NodeHandle ("~");
  
	v4r::DominantPlaneSegmenterParameter params;  
  segmenter_ = new v4r::DominantPlaneSegmenter<PointT>(params);
    
  Segment_ = n_->advertiseService ("/squirrel_segmentation", &SegmenterComplete::segment, this);
  ROS_INFO ("Squirrel_Segmentation_Server : Ready to get service calls...");
  ros::spin ();
}


int
main (int argc, char ** argv)
{
  SegmenterComplete m;
  m.initialize (argc, argv);

  return 0;
}
