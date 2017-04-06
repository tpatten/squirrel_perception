#ifndef ESF_NN_CLASSIFIER_H
#define ESF_NN_CLASSIFIER_H

#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/esf.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <fstream>
#include <iostream>
#include <dirent.h>

// Requires: sudo apt-get install libhdf5-dev

#define _VIEWS_DIR "views"
#define _ESF_DIR "esf"
#define _DESCRIPTOR_FILENAME "descriptor_"
#define _POSE_FILENAME "pose_"
#define _KDTREE_IDX_FILENAME "kdtree_esf.idx"
#define _H5_FILENAME "training_data_esf.h5"
#define _DATALIST_FILENAME "training_data_esf.list"
#define _NULL_STR "null"
#define _ESF_LENGTH 640

typedef pcl::PointXYZRGB PointT;

struct esf_model
{
    std::string _filename;
    std::string _class_name;
    std::string _pose_name;
    std::vector<float> _descriptor;

    friend std::ostream& operator<<(std::ostream& os, const esf_model& em)
    {
        os << "file: " << em._filename << ", class: " << em._class_name << ", pose: " << em._pose_name;
    }
};

class ESFNNClassifier
{
public:
    ESFNNClassifier();

    ESFNNClassifier(const ESFNNClassifier &other);

    ~ESFNNClassifier();

    ESFNNClassifier& operator=(const ESFNNClassifier &other);

    bool classify(std::vector<std::string> &class_names, std::vector<float> &probabilities, std::vector<std::string> &best_instances,
                  const std::vector<int> &indices = std::vector<int>());

    bool initialize();

    // Getters
    std::string get_model_dir() const;

    int get_NN() const;

    std::vector<std::string> get_class_names() const;

    std::vector<esf_model> get_train_data() const;

    flann::Matrix<float>* get_flann_data() const;

    flann::Index<flann::ChiSquareDistance<float> >* get_index() const;

    pcl::PointCloud<PointT> get_cloud() const;

    std::string get_kdtree_idx_file_name() const;

    std::string get_training_data_h5_file_name() const;

    std::string get_training_data_list_file_name() const;

    // Setters
    void set_model_directory(const std::string &model_dir);

    void set_NN(const int &NN);

    void set_cloud(const pcl::PointCloud<PointT> &cloud);

private:

    bool load_data();

    std::vector<std::string> get_subdirectories(const std::string &dir);

    std::vector<std::string> get_files(const std::string &dir, const std::string &prefix = _NULL_STR);

    bool build_tree();

    bool load_flann_data();

    bool load_files_list();

    bool load_esf_model(const std::string &filename, esf_model &em);

    std::vector<float> read_descriptor(const std::string &filename);

    esf_model compute_esf(const pcl::PointCloud<PointT> &cloud);

    bool compute_class_probabilities_and_instances(const flann::Matrix<int> &nn_indices,
                                                   std::vector<float> &probs, std::vector<std::string> &instances);

    std::string _model_dir;
    int _NN;
    std::vector<std::string> _class_names;
    std::vector<esf_model> _train_data;
    flann::Matrix<float> *_flann_data;
    flann::Index<flann::ChiSquareDistance<float> > *_index;
    pcl::PointCloud<PointT>::Ptr _cloud;

    std::string _kdtree_idx_file_name;// = train_dir + "kdtree_esf.idx";
    std::string _training_data_h5_file_name;// = train_dir + "training_data_esf.h5";
    std::string _training_data_list_file_name;// = train_dir + "training_data_esf.list";

};

#endif // ESF_NN_CLASSIFIER_H
