/*
 * squirrel_segmentation.hpp
 *
 *  Created on: Nov 7, 2014
 *      Author: Ekaterina Potapova
 */

#include <strstream>
#include <pcl/common/common.h>
#include <ros/ros.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
//#include <v4r/segmentation/dominant_plane_segmenter.h>
#include "squirrel_segmentation/dominant_plane_segmenter.h"
#include <squirrel_object_perception_msgs/Segment.h>

#ifndef SQUIRREL_SEGMENTATION_HPP_
#define SQUIRREL_SEGMENTATION_HPP_

class SegmenterComplete
{
private:
  typedef pcl::PointXYZRGB PointT;
  ros::ServiceServer Segment_;
  ros::NodeHandle *n_;
  //v4r::DominantPlaneSegmenter<PointT>* segmenter_;
	DominantPlaneSegmenter<PointT>* segmenter_;

  bool
  segment (squirrel_object_perception_msgs::Segment::Request &req, squirrel_object_perception_msgs::Segment::Response &response);
  
public:
  SegmenterComplete ();
  ~SegmenterComplete ();

  void
  initialize (int argc, char ** argv);
};

#endif //SQUIRREL_SEGMENTATION_HPP_
