#ifndef CLM_H_141021
#define CLM_H_141021

#include "detector.h"
#include "patch.h"
#include "shape.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>
#include <string>
#include <vector>

// Main Tracking Class
class Clm {
public:
    Clm();
    Clm(const std::string& data_dir, const std::string& cascade_file);
    Clm(const std::string& shape_file, const std::string& patch_file,
        const std::string& detector_file, const std::string& cascade_file);

    void init(const std::string& data_dir, const std::string& cascade_file);
    void init(const std::string& shape_file, const std::string& patch_file,
              const std::string& detector_file,
              const std::string& cascade_file);

    bool track(const cv::Mat& image, std::vector<cv::Point2f>& dst_points,
               const bool init_flag = false, const bool use_redetect = true);

    // Train and save shape, patch, detector
    static void train(const std::vector<std::string>& image_names,
                      const std::vector<std::vector<cv::Point2f> >& points_vecs,
                      const std::string& CASCADE_FILE,
                      const std::vector<int>& symmetry,
                      const std::vector<cv::Vec2i>& connections,
                      const std::string& OUTPUT_DIR);

    // Flip points using symmetry
    static void getFlippedPointsVecs(
        const std::vector<std::vector<cv::Point2f> >& src_vecs,
        std::vector<std::vector<cv::Point2f> >& dst_vecs,
        const std::vector<std::string>& image_names,
        const std::vector<int>& symmetry);

private:
    static const std::string SHAPE_FILE_NAME, PATCH_FILE_NAME,
        DETECTOR_FILE_NAME;

    static const int N_PATCH_SIZES;
    static const cv::Size PATCH_SIZES[];

    Shape shape;
    PatchContainer patch;
    Detector detector;

    // Previous frame points
    std::vector<cv::Point2f> pre_points;
};

#endif
