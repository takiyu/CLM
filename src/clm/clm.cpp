#include "clm.h"

using namespace cv;
using namespace std;

// Filenames
const string Clm::SHAPE_FILE_NAME = "shape.data";
const string Clm::PATCH_FILE_NAME = "patch.data";
const string Clm::DETECTOR_FILE_NAME = "detector.data";
// Patch sizes
const int Clm::N_PATCH_SIZES = 3;
const Size Clm::PATCH_SIZES[N_PATCH_SIZES] = {Size(21, 21), Size(11, 11),
                                              Size(5, 5)};

Clm::Clm() {}

Clm::Clm(const string& data_dir, const string& cascade_file) {
    init(data_dir, cascade_file);
}

Clm::Clm(const string& shape_file, const string& patch_file,
         const string& detector_file, const string& cascade_file) {
    init(shape_file, patch_file, detector_file, cascade_file);
}

void Clm::init(const string& data_dir, const string& cascade_file) {
    this->pre_points.clear();

    // Load
    this->shape.load(data_dir + "/" + SHAPE_FILE_NAME);
    this->patch.load(data_dir + "/" + PATCH_FILE_NAME);
    this->detector.load(data_dir + "/" + DETECTOR_FILE_NAME);
    this->detector.initCascade(cascade_file);
}

void Clm::init(const string& shape_file, const string& patch_file,
               const string& detector_file, const string& cascade_file) {
    this->pre_points.clear();

    // Load
    this->shape.load(shape_file);
    this->patch.load(patch_file);
    this->detector.load(detector_file);
    this->detector.initCascade(cascade_file);
}

bool Clm::track(const Mat& image, vector<Point2f>& result_points,
                const bool init_flag, const bool use_redetect) {
    //--- Initialize points ---
    if (init_flag == true || pre_points.size() == 0) {
        this->detector.detect(image, pre_points);
    }
    //--- Re-detect points (Translation) ---
    else {
        if (use_redetect) this->detector.redetect(image, pre_points);
    }
    // When failed to detect, exit
    if (pre_points.size() == 0) return false;

    //--- Track with each patch size ---
    for (int i = 0; i < N_PATCH_SIZES; i++) {
        // Patch search
        this->patch.calcPeaks(image, pre_points, pre_points, PATCH_SIZES[i]);
        // Express tracked points as a shape
        Mat shape_param = this->shape.getPraram(pre_points);
        this->shape.getShape(pre_points, shape_param);
    }

    // return the result
    result_points = pre_points;
    return true;
}

void Clm::train(const vector<string>& image_names,
                const vector<vector<Point2f> >& points_vecs,
                const string& CASCADE_FILE, const vector<int>& symmetry,
                const vector<Vec2i>& connections, const string& OUTPUT_DIR) {
    //====== Prepare flipped points ======
    vector<vector<Point2f> > flipped_points_vecs;
    getFlippedPointsVecs(points_vecs, flipped_points_vecs, image_names,
                         symmetry);

    //====== Shape ======
    Shape shape;
    // Unite the original points and the flipped ones
    vector<vector<Point2f> > united_points_vecs = points_vecs;
    united_points_vecs.insert(united_points_vecs.end(),
                              flipped_points_vecs.begin(),
                              flipped_points_vecs.end());
    // Train
    shape.train(united_points_vecs);
    // Save
    shape.save(OUTPUT_DIR + "/" + SHAPE_FILE_NAME);
    // Visualize
    shape.visualize(connections);

    // Get mean shape
    vector<Point2f> mean_shape;
    shape.getMeanShape(mean_shape, 100);

    //====== Patch ======
    // Train
    PatchContainer patch;
    patch.train(mean_shape, image_names, points_vecs, flipped_points_vecs);
    // Save
    patch.save(OUTPUT_DIR + "/" + PATCH_FILE_NAME);
    // Visualize
    patch.visualize();

    //====== Detector ======
    // Train
    Detector detector(CASCADE_FILE);
    detector.train(mean_shape, image_names, points_vecs, flipped_points_vecs);
    // Save
    detector.save(OUTPUT_DIR + "/" + DETECTOR_FILE_NAME);
    // Visualize
    detector.visualize();

    cout << "Training is finished" << endl;
}

void Clm::getFlippedPointsVecs(const vector<vector<Point2f> >& src_vecs,
                               vector<vector<Point2f> >& dst_vecs,
                               const vector<string>& image_names,
                               const vector<int>& symmetry) {
    assert(src_vecs.size() == image_names.size());

    // Initialize returning value
    dst_vecs.reserve(src_vecs.size());
    // Flip and register
    for (int i = 0; i < src_vecs.size(); i++) {
        // Get image width
        Mat tmp_mat = imread(image_names[i], 0);
        if (tmp_mat.empty()) {
            cerr << "Read Error : " << image_names[i] << endl;
            continue;
        }
        int width = tmp_mat.cols;

        // Flip
        vector<Point2f> flipped_points(src_vecs[i].size());
        assert(symmetry.size() == src_vecs[i].size());
        for (int j = 0; j < src_vecs[i].size(); j++) {
            flipped_points[j].x = width - 1 - src_vecs[i][symmetry[j]].x;
            flipped_points[j].y = src_vecs[i][symmetry[j]].y;
        }
        // Register
        dst_vecs.push_back(flipped_points);
    }
}
