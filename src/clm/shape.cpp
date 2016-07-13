#include "shape.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

/*
 * Point representation Mat(32FC1) is used only in shape class.
 * In out of shape class, vector<Point2f> is used.
*/

Shape::Shape() {}

void Shape::train(const vector<vector<Point2f> >& points_vecs,
                  const float EFF_EIGEN_PAR, const int MAX_EFF_IDX) {
    // Set point size and check input
    this->points_size = points_vecs[0].size();
    for (int i = 1; i < points_vecs.size(); i++) {
        assert(points_vecs[i].size() == this->points_size);
    }

    // Convert to inner representation Mat(32FC1)
    Mat orl_points_mat = cvtPointVecs2Mat32f(points_vecs);

    // Procrustes Analysis
    Mat points_mat = calcProcrustes(orl_points_mat);

    // Calculate rigid transformation's basis
    // (basis * param = points)
    Mat basis = calcRigidTransformationBasis(points_mat);

    // Get approximated parameter
    // (basis.t() * basis = Identity)
    Mat params = basis.t() * points_mat;
    // Representation error (the difference)
    Mat points_diff = points_mat - basis * params;

    // Eigen vectors to represent the difference
    Mat eigenvectors =
        calcEigenVectors(points_diff, EFF_EIGEN_PAR, MAX_EFF_IDX);

    // Combine basis and eigenvectors ( eff_idx == eigenvectors.cols )
    this->combiner.create(points_mat.rows, eigenvectors.cols + 4, CV_32F);
    basis.copyTo(this->combiner(Rect(0, 0, 4, basis.rows)));
    eigenvectors.copyTo(
        this->combiner(Rect(4, 0, eigenvectors.cols, eigenvectors.rows)));

    // Calculate parameter variance
    this->param_variance = this->calcParamVarience(orl_points_mat);

    return;
}

void Shape::getMeanShape(vector<Point2f>& points, const int width) {
    Mat mean_param = this->getPraram();
    mean_param.at<float>(0) = width / this->calcWidth();
    this->getShape(points, mean_param);
}

void Shape::getShape(vector<Point2f>& points, const Mat& param) {
    points.clear();
    points.reserve(this->points_size);
    // Calculate a shape from parameter
    Mat paramed_shape = this->combiner * param;
    // Mat -> vector<Point2f>
    for (int i = 0; i < paramed_shape.rows / 2; i++) {
        points.push_back(Point2f(paramed_shape.at<float>(2 * i),
                                 paramed_shape.at<float>(2 * i + 1)));
    }
}

Mat Shape::getPraram() { return Mat::zeros(this->combiner.cols, 1, CV_32F); }

Mat Shape::getPraram(const vector<Point2f> points) {
    assert(this->points_size == points.size());

    // vector<point2f> -> Mat
    Mat points_mat(2 * this->points_size, 1, CV_32F);
    // 1 point -> 1 row
    Mat y = Mat(points).reshape(1, 2 * this->points_size);
    y.copyTo(points_mat);

    Mat param = this->combiner.t() * points_mat;

    // Clamp using variance
    this->clampParam(param);

    return param;
}

float Shape::calcWidth() {
    // x coordinates
    float x_max = combiner.at<float>(0, 0);
    float x_min = combiner.at<float>(0, 0);
    for (int i = 1; i < combiner.rows / 2; i++) {
        x_max = max(x_max, combiner.at<float>(2 * i, 0));
        x_min = min(x_min, combiner.at<float>(2 * i, 0));
    }
    return x_max - x_min;
}

void Shape::save(const string& filename) {
    FileStorage cvfs(filename, CV_STORAGE_WRITE);
    write(cvfs, "Combiner", this->combiner);
    write(cvfs, "ParamVarience", this->param_variance);
}

void Shape::load(const string& filename) {
    FileStorage cvfs(filename, CV_STORAGE_READ);
    FileNode node(cvfs.fs, NULL);
    read(node["Combiner"], this->combiner);
    this->points_size = this->combiner.rows / 2;
    read(node["ParamVarience"], this->param_variance);
}

void Shape::visualize(const vector<Vec2i>& connections) {
    const Scalar COLOR(255);
    const string WINDOW_NAME = "visualized_shape";

    // Prepare parameter
    Mat param = this->getPraram();
    // Set scale and coordinate
    param.at<float>(0) = 200.0f / this->calcWidth();  // scale
    param.at<float>(2) = 1300;                        // dx
    param.at<float>(3) = 1300;                        // dy

    // Sequential variable
    vector<float> val;
    for (int i = 0; i < 50; i++) val.push_back(float(i) / 50);
    for (int i = 0; i < 50; i++) val.push_back(float(50 - i) / 50);
    for (int i = 0; i < 50; i++) val.push_back(-float(i) / 50);
    for (int i = 0; i < 50; i++) val.push_back(-float(50 - i) / 50);

    // Start to visualize
    std::cout << "press 'q' to exit visualize" << std::endl;
    while (true) {
        // scale and coordinates are static
        for (int n = 4; n < param.rows; n++) {
            for (int m = 0; m < val.size(); m++) {
                param.at<float>(n) = val[m] * 100;
                // Clamp parameter
                this->clampParam(param);
                // Get a shape from parameter
                vector<Point2f> paramed_points;
                this->getShape(paramed_points, param);

                // Draw and show
                Mat canvas = Mat::zeros(300, 300, CV_32F);
                Shape::drawPoints(canvas, paramed_points, COLOR, 1,
                                  connections);
                imshow(WINDOW_NAME, canvas);

                // Exit with 'q' key
                char key = waitKey(10);
                if (key == 'q') {
                    destroyWindow(WINDOW_NAME);
                    return;
                }
            }
        }
    }
}

void Shape::resizePoints(vector<Point2f>& points, float scale) {
    for (int i = 0; i < points.size(); i++) {
        points[i] *= scale;
    }
}

void Shape::shiftPoints(vector<Point2f>& points, Point2f shift) {
    for (int i = 0; i < points.size(); i++) {
        points[i] += shift;
    }
}

void Shape::drawPoints(Mat& image, const vector<Point2f>& points,
                       const Scalar& color, const int radius,
                       const vector<Vec2i>& connections) {
    // draw points
    for (int i = 0; i < points.size(); i++) {
        circle(image, points[i], radius, color, -1, 8, 0);
    }
    // draw lines
    for (int j = 0; j < connections.size(); j++) {
        line(image, points[connections[j][0]], points[connections[j][1]], color,
             radius);
    }
}

void Shape::drawPoints(Mat& image, const vector<Point2f>& points,
                       const Scalar& color, const int radius) {
    for (int i = 0; i < points.size(); i++) {
        circle(image, points[i], radius, color, -1, 8, 0);
        if (i != 0) line(image, points[i - 1], points[i], color, radius);
    }
    line(image, points[0], points[points.size() - 1], color, radius);
}

void Shape::drawPointsWithIdx(Mat& image, const vector<Point2f>& points,
                              const Scalar& color, const int radius) {
    drawPoints(image, points, color, radius);
    for (int i = 0; i < points.size(); i++) {
        stringstream ss;
        ss << i;
        // blue
        putText(image, ss.str(), points[i], FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                color);
    }
}

void Shape::drawPointsWithIdx(Mat& image, const vector<Point2f>& points,
                              const Scalar& color, const int radius,
                              const vector<Vec2i>& connections) {
    drawPoints(image, points, color, radius, connections);
    for (int i = 0; i < points.size(); i++) {
        stringstream ss;
        ss << i;
        // blue
        putText(image, ss.str(), points[i], FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                color);
    }
}

Mat Shape::cvtPointVecs2Mat32f(const vector<vector<Point2f> >& points_vecs) {
    int vec_size = points_vecs.size();
    assert(vec_size > 0);

    // Copy
    Mat dst(2 * this->points_size, vec_size, CV_32F);
    for (int i = 0; i < vec_size; i++) {
        //列に１データを保存
        Mat y = Mat(points_vecs[i]).reshape(1, 2 * this->points_size);
        y.copyTo(dst.col(i));
    }

    return dst;
}

Mat Shape::calcProcrustes(const Mat& points, const int max_times,
                          const float epsilon) {
    int vec_size = points.cols;

    // Move to the origin
    Mat dst_points = points.clone();
    for (int i = 0; i < vec_size; i++) {
        // Get mean point
        float mean_x = 0, mean_y = 0;
        for (int j = 0; j < this->points_size; j++) {
            mean_x += dst_points.at<float>(2 * j, i);
            mean_y += dst_points.at<float>(2 * j + 1, i);
        }
        mean_x /= this->points_size;
        mean_y /= this->points_size;
        // Subtract
        for (int j = 0; j < this->points_size; j++) {
            dst_points.at<float>(2 * j, i) -= mean_x;
            dst_points.at<float>(2 * j + 1, i) -= mean_y;
        }
    }

    // optimise scale and rotation
    Mat pre_mean_shape;
    for (int n = 0; n < max_times; n++) {
        // Get mean_shape
        Mat mean_shape = dst_points * Mat::ones(vec_size, 1, CV_32F) / vec_size;
        normalize(mean_shape, mean_shape);
        // if the error becomes low, exit
        if (n != 0) {
            if (norm(mean_shape, pre_mean_shape) < epsilon) break;
        }

        // Update previous mean shape
        pre_mean_shape = mean_shape.clone();

        for (int i = 0; i < vec_size; i++) {
            // Transformation matrix to mean_shape
            float a = 0, b = 0, d = 0;
            for (int j = 0; j < this->points_size; j++) {
                float src_0 = dst_points.at<float>(2 * j, i);
                float src_1 = dst_points.at<float>(2 * j + 1, i);
                float dst_0 = mean_shape.at<float>(2 * j, 0);
                float dst_1 = mean_shape.at<float>(2 * j + 1, 0);
                d += src_0 * src_0 + src_1 * src_1;
                a += src_0 * dst_0 + src_1 * dst_1;
                b += src_0 * dst_1 - src_1 * dst_0;
            }
            a /= d;
            b /= d;
            // matrix [a,-b; b,a]
            for (int j = 0; j < this->points_size; j++) {
                float x = dst_points.at<float>(2 * j, i);
                float y = dst_points.at<float>(2 * j + 1, i);
                dst_points.at<float>(2 * j, i) = a * x - b * y;
                dst_points.at<float>(2 * j + 1, i) = b * x + a * y;
            }
        }
    }
    return dst_points;
}

Mat Shape::calcRigidTransformationBasis(const Mat& src) {
    int vec_size = src.cols;
    Mat mean_shape = src * Mat::ones(vec_size, 1, CV_32F) / vec_size;

    // Basis
    Mat basis(2 * this->points_size, 4, CV_32F);
    for (int i = 0; i < this->points_size; i++) {
        basis.at<float>(2 * i, 0) = mean_shape.at<float>(2 * i);
        basis.at<float>(2 * i, 1) = -mean_shape.at<float>(2 * i + 1);
        basis.at<float>(2 * i, 2) = 1.0f;
        basis.at<float>(2 * i, 3) = 0.0f;

        basis.at<float>(2 * i + 1, 0) = mean_shape.at<float>(2 * i + 1);
        basis.at<float>(2 * i + 1, 1) = mean_shape.at<float>(2 * i);
        basis.at<float>(2 * i + 1, 2) = 0.0f;
        basis.at<float>(2 * i + 1, 3) = 1.0f;
    }

    // Gram-Schmidt orthogonalization
    for (int i = 0; i < 4; i++) {
        Mat v = basis.col(i);
        for (int j = 0; j < i; j++) {
            Mat w = basis.col(j);
            v -= w * (w.t() * v);  // v-=w * (dot)
        }
        normalize(v, v);
    }

    return basis;
}

Mat Shape::calcEigenVectors(const Mat& points_diff, const float EFF_EIGEN_PAR,
                            const int MAX_EFF_IDX) {
    // SVD (covariance matrix : diff*diff.t())
    SVD svd(points_diff * points_diff.t());

    // Sum up eigenvalues
    float eigenvalue_sum = 0;
    for (int i = 0; i < svd.w.rows; i++) {
        eigenvalue_sum += svd.w.at<float>(i, 0);
    }

    // decide valid dimension
    int max_idx =
        min(MAX_EFF_IDX, min(points_diff.cols - 1, this->points_size - 1));

    // effectual index (eff_idx)
    int eff_idx = 0;
    float tmp_sum = 0;
    for (; eff_idx < max_idx; eff_idx++) {
        tmp_sum += svd.w.at<float>(eff_idx, 0);
        // check the rate
        if (tmp_sum / eigenvalue_sum >= EFF_EIGEN_PAR) {
            // next
            if (eff_idx < max_idx - 1) eff_idx++;
            break;
        }
    }
    // Get eigenvectors (svd.u.rows == points_mat.rows)
    return svd.u(Rect(0, 0, eff_idx, svd.u.rows));
}

Mat Shape::calcParamVarience(const Mat& points_mat) {
    // Get parameters
    Mat params = this->combiner.t() * points_mat;
    // Normalize with the scaling
    for (int i = 0; i < params.cols; i++) {
        params.col(i) /= params.at<float>(0, i);
    }

    // variance matrix
    Mat variance(params.rows, 1, CV_32F);

    // square
    pow(params, 2, params);
    // matrix to calculate variance
    Mat ones_to_var =
        Mat::ones(1, params.cols, CV_32F) / (float)(params.cols - 1);
    // Calculate variance for each element
    for (int i = 0; i < variance.rows; i++) {
        if (i < 4)
            variance.at<float>(i) = -1;
        else
            variance.at<float>(i) = params.row(i).dot(ones_to_var);
    }

    return variance;
}

void Shape::clampParam(Mat& param, const float deviation_times) {
    float scale = param.at<float>(0);

    Mat param_abs = abs(param);
    // Only parameters
    for (int i = 4; i < this->param_variance.rows; i++) {
        float dev_x =
            scale * deviation_times * sqrtf(this->param_variance.at<float>(i));
        // Clamp
        if (param_abs.at<float>(i) > dev_x) {
            if (param.at<float>(i) > 0)
                param.at<float>(i) = dev_x;
            else
                param.at<float>(i) = -1 * dev_x;
        }
    }
}
