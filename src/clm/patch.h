#ifndef PATCH_H_141021
#define PATCH_H_141021

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

// One patch cell
class PatchCell {
public:
    PatchCell();

    void train(const std::vector<cv::Mat>& training_images,
               const cv::Size patch_size, const int Training_count = 1000,
               const float ideal_map_variance = 1.0,
               const float weight_init = 1e-3, const float train_fetter = 1e-6);

    // Calculate response for input image
    cv::Mat calcResponse(const cv::Mat& image);

    cv::Mat getPatch() { return this->patch.clone(); }
    void setPatch(const cv::Mat& src) { this->patch = src.clone(); }

    // Get patch size
    cv::Size getPatchSize() { return this->patch.size(); }

private:
    cv::Mat patch;

    // Convert to log image
    cv::Mat cvtLogImage(const cv::Mat& image);
};

// Patch cells container
class PatchContainer {
public:
    PatchContainer();

    // Train
    //  base_shape : assume mean shape
    //  flied_points_vecs : flipped points (If empty, it will be ignored)
    void train(const std::vector<cv::Point2f>& base_shape,
               const std::vector<std::string>& image_names,
               const std::vector<std::vector<cv::Point2f> >& points_vecs,
               const std::vector<std::vector<cv::Point2f> >& flied_points_vecs =
                   std::vector<std::vector<cv::Point2f> >(0),
               const cv::Size patch_size = cv::Size(11, 11),
               const cv::Size search_size = cv::Size(11, 11));
    // Calculate the most responsible points
    void calcPeaks(const cv::Mat& src_image,
                   const std::vector<cv::Point2f>& src_points,
                   std::vector<cv::Point2f>& dst_points,
                   const cv::Size search_size = cv::Size(21, 21));

    void save(const std::string& filename);
    void load(const std::string& filename);

    void visualize();

private:
    std::vector<cv::Point2f> base_shape;
    int points_size;  // ( == base_shape.size() )
    std::vector<PatchCell> patches;

    // Calculate affine matrix from base_shape to input points
    cv::Mat calcAffineFromBase(const std::vector<cv::Point2f>& points);
    // Update translation element of affine matrix to move the center of point
    void setAffineRotatedTranslation(cv::Mat& affine_mat,
                                     const cv::Point2f& point,
                                     const cv::Size& window_size);
    // Calculate inverse of affine matrix
    cv::Mat calcInverseAffine(const cv::Mat& affine_mat);
    // Apply affine matrix
    void applyAffineToPoints(const std::vector<cv::Point2f>& src_points,
                             const cv::Mat& aff_mat,
                             std::vector<cv::Point2f>& dst_points);
};

#endif
