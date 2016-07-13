#ifndef SHAPE_H_141017
#define SHAPE_H_141017

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

class Shape {
public:
    Shape();

    void train(const std::vector<std::vector<cv::Point2f> >& points_vecs,
               const float EFF_EIGEN_PAR = 0.95, const int MAX_EFF_IDX = 10);

    void getMeanShape(std::vector<cv::Point2f>& points, const int width = 100);

    // Get a shape using parameter
    void getShape(std::vector<cv::Point2f>& points, const cv::Mat& param);
    // Get empty parameter
    cv::Mat getPraram();
    // Get parameter from a shape
    cv::Mat getPraram(const std::vector<cv::Point2f> points);

    // Calculate width of the trained shape
    float calcWidth();

    void save(const std::string& filename);
    void load(const std::string& filename);

    void visualize(const std::vector<cv::Vec2i>& connections);

    // Scale points
    static void resizePoints(std::vector<cv::Point2f>& points, float scale);
    // Shift points
    static void shiftPoints(std::vector<cv::Point2f>& points,
                            cv::Point2f shift);

    /* Drawing */
    // Draw points using connections
    static void drawPoints(cv::Mat& image,
                           const std::vector<cv::Point2f>& points,
                           const cv::Scalar& color, const int radius,
                           const std::vector<cv::Vec2i>& connections);
    // Draw points
    static void drawPoints(cv::Mat& image,
                           const std::vector<cv::Point2f>& points,
                           const cv::Scalar& color, const int radius);
    // Draw points with its indices
    static void drawPointsWithIdx(cv::Mat& image,
                                  const std::vector<cv::Point2f>& points,
                                  const cv::Scalar& color, const int radius);
    // Draw points with its indices using connections
    static void drawPointsWithIdx(cv::Mat& image,
                                  const std::vector<cv::Point2f>& points,
                                  const cv::Scalar& color, const int radius,
                                  const std::vector<cv::Vec2i>& connections);

private:
    // the number of points ( = combiner.row / 2 )
    int points_size;
    // Shape matrix
    cv::Mat combiner;
    // parameter variance
    cv::Mat param_variance;

    // Convert Type: std::vector<cv::Point2f> -> 1-dim cv::Mat (32FC1)
    cv::Mat cvtPointVecs2Mat32f(
        const std::vector<std::vector<cv::Point2f> >& points_vecs);
    // Procrustes Analysis
    cv::Mat calcProcrustes(const cv::Mat& points, const int max_times = 100,
                           const float epsilon = 1e-6);
    // Calculate 2D rigid transformation's basis
    cv::Mat calcRigidTransformationBasis(const cv::Mat& src);
    // Calculate eigen vectors
    //	EFF_EIGEN_PAR	: valid eigen value rate
    //	MAX_EFF_IDX		: maximum dimension of eigen vectors
    cv::Mat calcEigenVectors(const cv::Mat& points_diff,
                             const float EFF_EIGEN_PAR, const int MAX_EFF_IDX);

    // Calculate parameter's variance
    cv::Mat calcParamVarience(const cv::Mat& points_mat);
    // Clamp parameter with deviation (default:3 sigma)
    void clampParam(cv::Mat& param, const float deviation_times = 3.0);
};

#endif
