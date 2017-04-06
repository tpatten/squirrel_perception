#include "squirrel_active_exploration/esf_nn_classifier.h"

using namespace std;
using namespace pcl;

ESFNNClassifier::ESFNNClassifier() : _NN (5), _cloud (new PointCloud<PointT>())
{}

ESFNNClassifier::ESFNNClassifier(const ESFNNClassifier &other)
{
    _model_dir = other.get_model_dir();
    _NN = other.get_NN();
    _class_names = other.get_class_names();
    _train_data = other.get_train_data();
    _flann_data = other.get_flann_data();
    _index = other.get_index();
    _kdtree_idx_file_name = other.get_kdtree_idx_file_name();
    _training_data_h5_file_name = other.get_training_data_h5_file_name();
    _training_data_list_file_name = other.get_training_data_list_file_name();
}

ESFNNClassifier::~ESFNNClassifier()
{
    delete[] _flann_data->ptr();
    delete _flann_data;
    delete _index;
}

ESFNNClassifier& ESFNNClassifier::operator=(const ESFNNClassifier &other)
{
    if (&other == this)
          return *this;

    _model_dir = other.get_model_dir();
    _NN = other.get_NN();
    _class_names = other.get_class_names();
    _train_data = other.get_train_data();
    _flann_data = other.get_flann_data();
    _index = other.get_index();
    _kdtree_idx_file_name = other.get_kdtree_idx_file_name();
    _training_data_h5_file_name = other.get_training_data_h5_file_name();
    _training_data_list_file_name = other.get_training_data_list_file_name();
}

bool ESFNNClassifier::classify(vector<string> &class_names, vector<float> &probabilities, vector<string> &best_instances,
                               const vector<int> &indices)
{
    if (_cloud->size() == 0)
    {
        ROS_ERROR("ESFNNClassifier::classify : input cloud is empty");
        return false;
    }

    // Extract the sub cloud
    PointCloud<PointT> subcloud;
    if (indices.size() > 0)
        copyPointCloud(*_cloud, indices, subcloud);
    else
        copyPointCloud(*_cloud, subcloud);

    ROS_INFO("ESFNNClassifier::classify : classifying cloud of size %lu", subcloud.size());

    // Compute the ESF feature
    esf_model histogram = compute_esf(subcloud);
    // Convert to flann::Matrix
    flann::Matrix<float> p = flann::Matrix<float>(new float[histogram._descriptor.size()], 1, histogram._descriptor.size());
    memcpy (&p.ptr ()[0], &histogram._descriptor[0], p.cols*p.rows*sizeof (float));

    // Compute the _NN nearest neighbours
    flann::Matrix<int> nn_indices = flann::Matrix<int>(new int[_NN], 1, _NN);
    flann::Matrix<float> nn_distances = flann::Matrix<float>(new float[_NN], 1, _NN);
    _index->knnSearch(p, nn_indices, nn_distances, _NN, flann::SearchParams (512));
    delete[] p.ptr();

    // Compute probabilities
    class_names = _class_names;
    if (!compute_class_probabilities_and_instances(nn_indices, probabilities, best_instances))
    {
        ROS_ERROR("ESFNNClassifier::classify : could not compute class probabilities");
        return false;
    }

    cout << " * * * Nearest neighbours * * *" << endl;
    for (int i = 0; i < _NN; ++i)
        cout << _train_data[nn_indices[0][i]]._filename << " (" << nn_distances[0][i] << ")" << endl;
    cout << " * * * Class probabilities * * *" << endl;
    for (size_t i = 0; i < class_names.size(); ++i)
        cout << class_names[i] << " " << probabilities[i] << " " << best_instances[i] << endl;

    return true;
}

bool ESFNNClassifier::initialize()
{
    // Check has valid model directory
    if (_model_dir.size() == 0)
    {
        ROS_ERROR("ESFNNClassifier::initialize : model_dir is not set!");
        return false;
    }

    // Load flann data
    _kdtree_idx_file_name = _model_dir + "/" + _KDTREE_IDX_FILENAME;
    _training_data_h5_file_name = _model_dir + "/" + _H5_FILENAME;
    _training_data_list_file_name = _model_dir + "/" + _DATALIST_FILENAME;
    if (!load_flann_data())
    {
        ROS_WARN("ESFNNClassifier::initialize : could not load flann data, building it now ...");
        // Load the training data
        if (!load_data())
        {
            ROS_ERROR("ESFNNClassifier::initialize : could not load data");
            return false;
        }
        if (_train_data.size() <= 0)
        {
            ROS_ERROR("ESFNNClassifier::initialize : _train_data has size %lu", _train_data.size());
            return false;
        }
        // Build the tree and save to file
        if (!build_tree())
        {
            ROS_ERROR("ESFNNClassifier::initialize : failed to build tree");
            return false;
        }
        //flann::Matrix<float> flann_data (new float[_train_data.size() * _train_data[0]._descriptor.size()],
        //                                     _train_data.size(), _train_data[0]._descriptor.size());
    }
    // And now build the index
    //flann::Index<flann::ChiSquareDistance<float> > index (_flann_data, flann::SavedIndexParams(_kdtree_idx_file_name));
    //index.buildIndex();

    return true;
}

string ESFNNClassifier::get_model_dir() const { return _model_dir; }

int ESFNNClassifier::get_NN() const { return _NN; }

vector<string> ESFNNClassifier::get_class_names() const { return _class_names; }

vector<esf_model> ESFNNClassifier::get_train_data() const { return _train_data; }

flann::Matrix<float>* ESFNNClassifier::get_flann_data() const { return _flann_data; }

flann::Index<flann::ChiSquareDistance<float> >* ESFNNClassifier::get_index() const { return _index; }

PointCloud<PointT> ESFNNClassifier::get_cloud() const { return *_cloud; }

string ESFNNClassifier::get_kdtree_idx_file_name() const { return _kdtree_idx_file_name; }

string ESFNNClassifier::get_training_data_h5_file_name() const { return _training_data_h5_file_name; }

string ESFNNClassifier::get_training_data_list_file_name() const { return _training_data_list_file_name; }

void ESFNNClassifier::set_model_directory(const string &model_dir) { _model_dir = model_dir; }

void ESFNNClassifier::set_NN(const int &NN) { _NN = NN; }

void ESFNNClassifier::set_cloud(const PointCloud<PointT> &cloud) { copyPointCloud(cloud, *_cloud); }

/* Private Functions */

bool ESFNNClassifier::load_data()
{
    /* Load all esf descriptors in the hierarchy
       model_dir
           |
           -- class 1
                 |
                 -- instance 1
                        |
                        -- esf
                            |
                            -- decriptor 1
                            .
                            -- descriptor N
                 .
                 -- instance N
           .
           -- class N
    */

    _class_names.clear();

    // Get the class names
    vector<string> class_names = get_subdirectories(_model_dir);
    if (class_names.size() == 0)
    {
        ROS_ERROR("ESFNNClassifier::load_data : did not find any subdirectories in %s", _model_dir.c_str());
        return false;
    }

    // Load the instances
    for (vector<string>::const_iterator it1 = class_names.begin(); it1 != class_names.end(); ++it1)
    {
        string class_dir = _model_dir + "/" + *it1;
        vector<string> instance_names = get_subdirectories(class_dir);
        if (instance_names.size() == 0)
        {
            ROS_ERROR("ESFNNClassifier::load_data : did not find any subdirectories in %s", class_dir.c_str());
        }
        else
        {
            // Add this class
            _class_names.push_back(*it1);
            //cout << "-- Class " << *it1 << " --" << endl;
            // Load each descriptor for each instance
            for (vector<string>::const_iterator it2 = instance_names.begin(); it2 != instance_names.end(); ++it2)
            {
                //cout << "- Instance " << *it2 << " -" << endl;
                string instance_dir = class_dir + "/" + *it2 + "/" + _ESF_DIR;
                vector<string> descriptor_files = get_files(instance_dir, _DESCRIPTOR_FILENAME);
                for (vector<string>::const_iterator it3 = descriptor_files.begin(); it3 != descriptor_files.end(); ++it3)
                {
                    string f = instance_dir + "/" + *it3;
                    vector<float> desc = read_descriptor(f);
                    if (desc.size() == _ESF_LENGTH)
                    {
                        esf_model em;
                        em._descriptor = desc;
                        em._filename = f;
                        em._class_name = *it1;
                        // Get the number from the descriptor filename
                        size_t underscore = it3->find("_");
                        size_t ext = it3->find(".txt");
                        if (underscore != string::npos && ext != string::npos && (ext-underscore) >= 1)
                        {
                            string id = it3->substr(underscore+1, ext-underscore);
                            em._pose_name = class_dir + "/" + *it2 + "/" + _VIEWS_DIR + "/" + _POSE_FILENAME + id + ".txt";
                            cout << "Storing " << f << endl;
                            _train_data.push_back(em);
                        }
                    }

                }
            }
        }
    }

    ROS_INFO("ESFNNClassifier::load_data : found %lu training elements and %lu class names", _train_data.size(), _class_names.size());
    for (size_t i = 0; i < _class_names.size(); ++i)
        cout << _class_names[i] << endl;

    return true;
}

vector<string> ESFNNClassifier::get_subdirectories(const string &dir)
{
    vector<string> subdirs;

    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        ROS_ERROR("ESFNNClassifier::get_subdirectories : could not open %s", dir.c_str());
        return subdirs;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        string f = string(dirp->d_name);
        if (strcmp(f.c_str(),".") != 0 && strcmp(f.c_str(),"..") != 0 && f.find(".") == string::npos)
            subdirs.push_back(f);
    }

    return subdirs;
}

vector<string> ESFNNClassifier::get_files(const string &dir, const string &prefix)
{
    // Get all files begining with the prefix
    vector<string> files;

    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        ROS_ERROR("ESFNNClassifier::get_files : could not open %s", dir.c_str());
        return files;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        string f = string(dirp->d_name);
        // If it is not the . or .. directory
        if (strcmp(f.c_str(),".") != 0 && strcmp(f.c_str(),"..") != 0)
        {
            // Must contain a .txt or .pcd
            if (f.find(".txt") != string::npos || f.find(".pcd") != string::npos)
            {
                // Must contain the prefix
                if (prefix != _NULL_STR || f.find(prefix) != string::npos)
                    files.push_back(f);
            }
        }
    }

    return files;
}

bool ESFNNClassifier::build_tree()
{
    if (_train_data.size() <= 0)
    {
        ROS_ERROR("ESFNNClassifier::build_tree : _train_data has size %lu", _train_data.size());
        return false;
    }

    // Convert data into FLANN format
    //flann::Matrix<float> data (new float[_train_data.size() * _train_data[0]._descriptor.size()],
    //                                     _train_data.size(), _train_data[0]._descriptor.size());
    //_flann_data = data;
    _flann_data = new flann::Matrix<float>(new float[_train_data.size() * _train_data[0]._descriptor.size()],
                                                     _train_data.size(), _train_data[0]._descriptor.size());

    for (size_t i = 0; i < _flann_data->rows; ++i)
    {
        for (size_t j = 0; j < _flann_data->cols; ++j)
            (*_flann_data)[i][j] = _train_data[i]._descriptor[j];
    }

    // Save data to disk (list of models)
    flann::save_to_file (*_flann_data, _training_data_h5_file_name, "training_data");
    ofstream fs;
    fs.open (_training_data_list_file_name.c_str());
    for (size_t i = 0; i < _train_data.size (); ++i)
        fs << _train_data[i]._filename << "\n";
    fs.close();

    // Build the tree index and save it to disk
    pcl::console::print_error ("Building the kdtree index (%s) for %d elements...\n", _kdtree_idx_file_name.c_str (), (int)_flann_data->rows);
    //flann::Index<flann::ChiSquareDistance<float> > index (*_flann_data, flann::KDTreeIndexParams(4));
    //index.buildIndex();
    //index.save (_kdtree_idx_file_name);
    _index = new flann::Index<flann::ChiSquareDistance<float> >(*_flann_data, flann::KDTreeIndexParams(4));
    _index->buildIndex();
    _index->save (_kdtree_idx_file_name);

    return true;
}

bool ESFNNClassifier::load_flann_data()
{
    // Check if the data has already been saved to disk
    if (!boost::filesystem::exists (_training_data_h5_file_name) || !boost::filesystem::exists (_training_data_list_file_name))
    {
        ROS_WARN("ESFNNClassifier::load_flann_data : could not find training data models files %s and %s",
                  _training_data_h5_file_name.c_str (), _training_data_list_file_name.c_str ());
        return false;
    }
    else
    {
        if (!load_files_list())
        {
            ROS_WARN("ESFNNClassifier::load_flann_data : could not load files list");
            return false;
        }
        _flann_data = new flann::Matrix<float>(new float[_train_data.size() * _train_data[0]._descriptor.size()],
                                                         _train_data.size(), _train_data[0]._descriptor.size());
        flann::load_from_file (*_flann_data, _training_data_h5_file_name, "training_data");
        ROS_INFO("ESFNNClassifier::load_flann_data : training data found, loaded %d ESF models from %s and %s.\n",
                 (int)_flann_data->rows, _training_data_h5_file_name.c_str(), _training_data_list_file_name.c_str());
    }

    // Check if the tree index has already been saved to disk
    if (!boost::filesystem::exists (_kdtree_idx_file_name))
    {
        ROS_WARN("ESFNNClassifier::load_flann_data : could not find kd-tree index in file %s!", _kdtree_idx_file_name.c_str ());
        return false;
    }

    _index = new flann::Index<flann::ChiSquareDistance<float> > (*_flann_data, flann::SavedIndexParams(_kdtree_idx_file_name));
    _index->buildIndex();

    ROS_INFO("ESFNNClassifier::load_flann_data : found %lu training elements and %lu class names", _train_data.size(), _class_names.size());
    for (size_t i = 0; i < _class_names.size(); ++i)
        cout << _class_names[i] << endl;

    return true;
}

bool ESFNNClassifier::load_files_list()
{
    _class_names.clear();
    _train_data.clear();

    ifstream fs;
    fs.open (_training_data_list_file_name.c_str ());
    if (!fs.is_open () || fs.fail ())
        return false;

    string line;
    while (!fs.eof ())
    {
        getline (fs, line);
        if (line.empty ())
            continue;
        esf_model em;
        if (!load_esf_model(line, em))
        {
            ROS_ERROR("ESFNNClassifier::load_files_list : cannot process file %s", line.c_str());
            return false;
        }
        _train_data.push_back (em);
        // Add to class names
        if (_class_names.size() == 0)
        {
            _class_names.push_back(em._class_name);
        }
        else
        {
            // Does this class name already exist in the list
            if (find(_class_names.begin(), _class_names.end(), em._class_name) == _class_names.end())
                _class_names.push_back(em._class_name);
        }
    }
    fs.close ();
    return true;
}

bool ESFNNClassifier::load_esf_model(const string &filename, esf_model &em)
{
    /*
      File name structure is: _model_dir/class/instance/esf/descriptor_x.txt
      Want:
        std::string _filename;
        std::string _class_name;
        std::string _pose_name;
        std::vector<float> _descriptor;
   */

    // Remove the model directory from the string
    if (filename.size() < _model_dir.size())
    {
        ROS_ERROR("ESFNNClassifier::load_esf_model : model directory %s does not exist in filename %s", _model_dir.c_str(), filename.c_str());
        return false;
    }
    string str = filename.substr(_model_dir.size()+1);
    // Extract the class name
    size_t slash = str.find_first_of("/");
    string class_name = str.substr(0, slash);
    str = str.substr(slash);
    // Now replace "esf" with "views" and "descriptor" with "pose"
    string pose_filename = filename;
    if (pose_filename.find(_ESF_DIR) != string::npos)
    {
        boost::replace_all(pose_filename, _ESF_DIR, _VIEWS_DIR);
    }
    else
    {
        ROS_ERROR("ESFNNClassifier::load_esf_model : filename %s does not contain subdirectory %s", filename.c_str(), _ESF_DIR);
        return false;
    }
    if (pose_filename.find(_DESCRIPTOR_FILENAME) != string::npos)
    {
        boost::replace_all(pose_filename, _DESCRIPTOR_FILENAME, _POSE_FILENAME);
    }
    else
    {
        ROS_ERROR("ESFNNClassifier::load_esf_model : filename %s does not contain %s", filename.c_str(), _DESCRIPTOR_FILENAME);
        return false;
    }
    // Create the esf model
    em._filename = filename;
    em._class_name = class_name;
    em._pose_name = pose_filename;
    em._descriptor = read_descriptor(filename);
    if (em._descriptor.size() != _ESF_LENGTH)
    {
        ROS_ERROR("ESFNNClassifier::load_esf_model : failed to read descriptor");
        return false;
    }

    return true;
}

vector<float> ESFNNClassifier::read_descriptor(const string &filename)
{
    fstream fp;
    fp.open(filename.c_str());

    vector<float> result;
    float number;
    if (fp.is_open())
    {
        while (fp >> number)
        {
            result.push_back(number);
            fp.get();
        }
    }

    return result;
}

esf_model ESFNNClassifier::compute_esf(const pcl::PointCloud<PointT> &cloud)
{
    ESFEstimation<PointT, ESFSignature640> esf;
    esf_model desc;
    PointCloud<ESFSignature640> ESF_signature;
    PointCloud<PointT>::Ptr cloud_ptr (new PointCloud<PointT>(cloud));
    esf.setInputCloud(cloud_ptr);
    esf.compute (ESF_signature);
    desc._descriptor.resize(_ESF_LENGTH);
    for(int i = 0; i < _ESF_LENGTH; ++i)
        desc._descriptor[i] = (float)*(ESF_signature.points[0].histogram+i);

    return desc;
}

bool ESFNNClassifier::compute_class_probabilities_and_instances(const flann::Matrix<int> &nn_indices,
                                                                vector<float> &probs, vector<string> &instances)
{
    // Count the number of instances for each class
    vector<int> counts;
    counts.resize(_class_names.size());
    instances.resize(_class_names.size());
    for (size_t i = 0; i < _class_names.size(); ++i)
    {
        counts[i] = 0;
        instances[i] = _NULL_STR;
    }

    string str;
    int sum = 0;
    for (int i = 0; i < _NN; ++i)
    {
        str = _train_data.at(nn_indices[0][i])._class_name;
        // Increment the count matching this
        if (find(_class_names.begin(), _class_names.end(), str) != _class_names.end())
        {
            int pos = distance(_class_names.begin(), find(_class_names.begin(), _class_names.end(), str));
            ++counts[pos];
            ++sum;
            // If this was the first instance for this class (then this is the best pose)
            if (counts[pos] == 1)
                instances[pos] = _train_data.at(nn_indices[0][i])._pose_name;
        }
    }

    // Normalise the counts to get a probability
    probs.resize(_class_names.size());
    for (size_t i = 0; i < counts.size(); ++i)
        probs[i] = (double)counts[i] / (double)sum;

    return true;
}
