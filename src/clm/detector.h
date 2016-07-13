#ifndef DETECTOR_H_141021
#define DETECTOR_H_141021

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>
#include <string>
#include <vector>

class Detector {
public:
    Detector();
    Detector(const std::string& cascade_file);

    // Initialize cascade
    void initCascade(const std::string& cascade_file);

    // Train
    void train(const std::vector<cv::Point2f>& base_shape,
               const std::vector<std::string>& image_names,
               const std::vector<std::vector<cv::Point2f> >& points_vecs,
               const std::vector<std::vector<cv::Point2f> >& flied_points_vecs =
                   std::vector<std::vector<cv::Point2f> >(0),
               const float cascade_scale = 1.1,
               const int cascade_min_neighbours = 2,
               const cv::Size cascade_min_size = cv::Size(30, 30),
               const float bounding_per = 0.8);

    // Detect (When failed, dst_points.size() is 0)
    void detect(const cv::Mat& image, std::vector<cv::Point2f>& dst_points,
                const float cascade_scale = 1.1,
                const int cascade_min_neighbours = 2,
                const cv::Size cascade_min_size = cv::Size(30, 30));
    // Re-detect (translation)
    void redetect(const cv::Mat& image, std::vector<cv::Point2f>& points);

    void save(const std::string& filename);
    void load(const std::string& filename);

    // Visualize (using webcam)
    void visualize();
    // Visualize (using image files)
    void visualize(const std::vector<std::string>& image_names);

private:
    cv::CascadeClassifier cascade;

    // Face scale to speed up for re-detection
    static const float SMALL_IMAGE_SCALE;

    std::vector<cv::Point2f> base_shape;
    cv::Vec3f offsets;  // offset between the rectangle and base shape
    cv::Rect face_rect;
    cv::Mat face_small, pre_face_small;

    // Check if points are valid
    bool isBoundingEnough(const std::vector<cv::Point2f>& points,
                          const cv::Rect& rect, const float percent);
    // Calculate mass center of points
    cv::Point2f calcMassCenter(const std::vector<cv::Point2f>& points);
    // Calculate scale for the base shape
    float calcScaleForBase(const std::vector<cv::Point2f>& points);
};

#endif
