#include <pcl/apps/dominant_plane_segmentation.h>
#include <pcl/impl/instantiate.hpp>
#include <boost/pointer_cast.hpp>
#include "squirrel_segmentation/dominant_plane_segmenter.h"

/*
template<typename PointT>
void
DominantPlaneSegmenter<PointT>::segment()
{
    clusters_.clear();
    pcl::apps::DominantPlaneSegmentation<PointT> dps;
    typename pcl::PointCloud<PointT>::Ptr scene ( boost::const_pointer_cast< pcl::PointCloud<PointT> > (scene_ ) ); ///NOTE: This cast is due to an PCL issue!
    dps.setInputCloud ( scene );
    dps.setMaxZBounds (param_.chop_z_);
    dps.setObjectMinHeight (param_.object_min_height_);
    dps.setObjectMaxHeight (param_.object_max_height_);
    dps.setMinClusterSize (param_.min_cluster_size_);
    dps.setWSize (param_.w_size_px_);
    dps.setDistanceBetweenClusters (param_.min_distance_between_clusters_);
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    dps.setDownsamplingSize ( param_.downsampling_size_ );
    if(param_.compute_table_plane_only_)
    {
        dps.compute_table_plane();
    }
    else
    {
        dps.compute_fast (clusters);
        dps.getIndicesClusters (clusters_);
    }
    dps.getTableCoefficients (dominant_plane_);
}
*/

/*
template<typename PointT>
void
DominantPlaneSegmenter<PointT>::getSegmentIndices(std::vector<pcl::PointIndices> &indices)
{
		indices = _indices;
}*/

PCL_INSTANTIATE(DominantPlaneSegmenter, PCL_XYZ_POINT_TYPES )
