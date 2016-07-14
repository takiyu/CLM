#ifndef APPEARANCE_H_141129
#define APPEARANCE_H_141129

#include <opencv2/core/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

class Appearance {
public:
    Appearance();
    Appearance(const std::vector<cv::Point2f> init_shape);

    void init(const std::vector<cv::Point2f> init_shape);

    void warp(const cv::Mat& src_im, cv::Mat& dst_im,
              const std::vector<cv::Point2f>& src_points,
              const std::vector<cv::Point2f>& dst_points);

private:
    std::vector<std::vector<int> > triangle_map;

    void makeTriangleIdxMap(const std::vector<cv::Point2f>& src_points,
                            std::vector<std::vector<int> >& tri_idxs);

    void warpTriangle(const cv::Mat& src_image, cv::Mat& dst_image,
                      const cv::Point2f src_tri[], const cv::Point2f dst_tri[]);
};

#endif
