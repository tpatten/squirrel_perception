/******************************************************************************
 * Copyright (c) 2015 Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/
/**
*
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date April, 2016
*      @brief dominant plane segmentation (taken from PCL)
*/
#pragma once
#include <pcl/apps/dominant_plane_segmentation.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/pointer_cast.hpp>
#include <boost/program_options.hpp>
#include "boost/date_time/posix_time/posix_time_types.hpp"
#include <glog/logging.h>
#include <v4r/core/macros.h>
#include <v4r/segmentation/segmenter.h>
namespace po = boost::program_options;

class DominantPlaneSegmenterParameter
{
public:
    int min_cluster_size_;
    float object_min_height_;
    float object_max_height_;
    float chop_z_;
    float min_distance_between_clusters_;
    int w_size_px_;
    float downsampling_size_;
    bool compute_table_plane_only_;
    DominantPlaneSegmenterParameter (int min_cluster_size=500,
               float object_min_height = 0.01f,
               float object_max_height = 0.7f,
               float chop_at_z = 3.f,
               float min_distance_between_clusters = 0.03f,
               int w_size_px = 5,
               float downsampling_size = 0.005f,
               bool compute_table_plane_only = false
            )
        :
          min_cluster_size_ (min_cluster_size),
          object_min_height_ (object_min_height),
          object_max_height_ (object_max_height),
          chop_z_ (chop_at_z),
          min_distance_between_clusters_ (min_distance_between_clusters),
          w_size_px_ (w_size_px),
          downsampling_size_ (downsampling_size),
          compute_table_plane_only_ ( compute_table_plane_only )
    {}
    /**
     * @brief init parameters
     * @param command_line_arguments (according to Boost program options library)
     * @return unused parameters (given parameters that were not used in this initialization call)
     */
    std::vector<std::string>
    init(int argc, char **argv)
    {
            std::vector<std::string> arguments(argv + 1, argv + argc);
            return init(arguments);
    }
    /**
     * @brief init parameters
     * @param command_line_arguments (according to Boost program options library)
     * @return unused parameters (given parameters that were not used in this initialization call)
     */
    std::vector<std::string>
    init(const std::vector<std::string> &command_line_arguments)
    {
        po::options_description desc("Dominant Plane Segmentation\n=====================");
        desc.add_options()
                ("help,h", "produce help message")
                ("seg_min_cluster_size", po::value<int>(&min_cluster_size_)->default_value(min_cluster_size_), "")
                ("seg_obj_min_height", po::value<float>(&object_min_height_)->default_value(object_min_height_), "")
                ("seg_obj_max_height", po::value<float>(&object_max_height_)->default_value(object_max_height_), "")
                ("seg_chop_z", po::value<float>(&chop_z_)->default_value(chop_z_), "")
                ("seg_min_distance_between_clusters", po::value<float>(&min_distance_between_clusters_)->default_value(min_distance_between_clusters_), "")
                ("seg_w_size_px", po::value<int>(&w_size_px_)->default_value(w_size_px_), "")
                ("seg_downsampling_size", po::value<float>(&downsampling_size_)->default_value(downsampling_size_), "")
                ("seg_compute_table_plane_only", po::value<bool>(&compute_table_plane_only_)->default_value(compute_table_plane_only_), "if true, only computes the table plane and not the Euclidean clusters. This should be faster.")
                ;
        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
        std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
        po::store(parsed, vm);
        if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
        try { po::notify(vm); }
        catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
        return to_pass_further;
    }
};

template<typename PointT>
class DominantPlaneSegmenter : public v4r::Segmenter<PointT>
{
    //using v4r::Segmenter<PointT>::indices_;
		std::vector<pcl::PointIndices> indices_;
    using v4r::Segmenter<PointT>::normals_;
    using v4r::Segmenter<PointT>::clusters_;
    using v4r::Segmenter<PointT>::scene_;
    //using v4r::Segmenter<PointT>::dominant_plane_;
		Eigen::Vector4f dominant_plane_;
    DominantPlaneSegmenterParameter param_;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DominantPlaneSegmenter(const DominantPlaneSegmenterParameter &p = DominantPlaneSegmenterParameter() ) :
        param_(p)
    {}
    bool getRequiresNormals() { return false; }
		void getSegmentIndices(std::vector<pcl::PointIndices> &indices) { indices = indices_; }

		void segment()
		{
				boost::posix_time::ptime t_start(boost::posix_time::microsec_clock::local_time());

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
				    //dps.getIndicesClusters (clusters_); // No longer in PCL 1.7!!

						indices_.clear();
						indices_.resize(clusters.size());
						pcl::KdTreeFLANN<PointT> kdtree;
  					kdtree.setInputCloud (scene);
						int K = 1;
  					std::vector<int> pointIdxNKNSearch(K);
  					std::vector<float> pointNKNSquaredDistance(K);
						for (size_t i = 0; i < clusters.size(); ++i)
						{
								// Extract the indices in the original point cloud
								for (size_t j = 0; j < clusters[i]->size(); ++j)
								{
										if ( kdtree.nearestKSearch (clusters[i]->points[j], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
										{
												if (pointNKNSquaredDistance[0] < 0.1)
												{
														indices_[i].indices.push_back(pointIdxNKNSearch[0]);
												}
										}
								}
						}
				}
				dps.getTableCoefficients (dominant_plane_);

				boost::posix_time::ptime t_end(boost::posix_time::microsec_clock::local_time());
    		boost::posix_time::time_duration duration(t_end - t_start);
    		std::cout << "Segmentation took " << duration.total_milliseconds() << " ms" << std::endl;
		}

    typedef boost::shared_ptr< DominantPlaneSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< DominantPlaneSegmenter<PointT> const> ConstPtr;
};
