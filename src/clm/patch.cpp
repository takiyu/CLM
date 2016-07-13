#include "patch.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

PatchCell::PatchCell() {}

void PatchCell::train(const vector<Mat>& training_images, const Size patch_size,
                      const int training_count, const float ideal_map_variance,
                      const float weight_init, const float train_fetter) {
    int images_num = training_images.size();
    int patch_pixel_num = patch_size.width * patch_size.height;

    // Check the size of image and patch size
    Size training_size = training_images[0].size();
    assert(training_size.width >= patch_size.width);
    assert(training_size.height >= patch_size.height);

    // Create ideal response map
    int map_width = training_size.width - patch_size.width;
    int map_height = training_size.height - patch_size.height;
    Mat ideal_map(map_width, map_height, CV_32F);
    // Gaussian distribution
    for (int y = 0; y < map_height; y++) {
        float dist_y = (map_height - 1) / 2 - y;
        for (int x = 0; x < map_width; x++) {
            float dist_x = (map_width - 1) / 2 - x;
            ideal_map.at<float>(y, x) =
                exp(-1 * (dist_x * dist_x + dist_y * dist_y) /
                    (ideal_map_variance * 2));
        }
    }
    // Normalize [0:1]
    normalize(ideal_map, ideal_map, 0, 1, NORM_MINMAX);

    // Initialize patch
    this->patch = Mat::zeros(patch_size, CV_32F);
    // Patch difference for update
    Mat patch_diff(patch_size, CV_32F);
    // Matrix to calculate average value of training patch
    Mat ones_to_avg = Mat::ones(patch_size, CV_32F) / patch_pixel_num;

    // Current training weight
    float weight = weight_init;
    // Training weight rate
    float weight_rate = pow(1e-8 / weight, 1.0 / training_count);

    // Random number generator for image indices
    RNG random(getTickCount());

    // Start training
    for (int n = 0; n < training_count; n++) {
        // Get random image
        int i = random.uniform(0, images_num);
        // Convert to log image
        Mat log_image = this->cvtLogImage(training_images[i]);

        // Calculate patch difference to update
        patch_diff = 0.0;
        for (int y = 0; y < map_height; y++) {
            for (int x = 0; x < map_width; x++) {
                // Extract training region
                Mat training_patch =
                    log_image(Rect(x, y, patch_size.width, patch_size.height))
                        .clone();
                // Subtract the average
                training_patch -= training_patch.dot(ones_to_avg);
                normalize(training_patch, training_patch);

                // Real and ideal responses
                float real_response_dot = this->patch.dot(training_patch);
                float ideal_respoinse_dot = ideal_map.at<float>(y, x);
                // Update patch difference using error rate
                patch_diff +=
                    (ideal_respoinse_dot - real_response_dot) * training_patch;
            }
        }

        // Update patch with normalize term
        this->patch += weight * (patch_diff - train_fetter * this->patch);
        // Update weight
        weight *= weight_rate;

        // Debug Visualize (same to calcResponse())
        // Mat response;
        // matchTemplate(log_image, this->patch, response, CV_TM_CCOEFF_NORMED);
        // Mat normed_patch;
        // normalize(this->patch, normed_patch, 0, 1, NORM_MINMAX);
        // normalize(patch_diff, patch_diff, 0, 1, NORM_MINMAX);
        // normalize(response, response, 0, 1, NORM_MINMAX);
        // imshow("patch", normed_patch);
        // imshow("patch_diff", patch_diff);
        // imshow("response", response);

        if (waitKey(10) == 'q') break;
    }
    return;
}

Mat PatchCell::calcResponse(const Mat& image) {
    Mat log_image = this->cvtLogImage(image);

    Mat response;
    matchTemplate(log_image, this->patch, response, CV_TM_CCOEFF_NORMED);

    // Make the sum 1
    normalize(response, response, 0, 1, NORM_MINMAX);
    response /= sum(response)[0];

    return response;
}

Mat PatchCell::cvtLogImage(const Mat& image) {
    Mat dst;

    // Convert to gray image
    if (image.channels() == 3)
        cvtColor(image, dst, CV_RGB2GRAY);
    else
        dst = image;
    assert(dst.channels() == 1);

    // Convert to float image
    if (dst.type() != CV_32F) dst.convertTo(dst, CV_32F);

    // Log
    dst += 1.0;
    log(dst, dst);

    return dst;
}

PatchContainer::PatchContainer() {}

void PatchContainer::train(const vector<Point2f>& base_shape,
                           const vector<string>& image_names,
                           const vector<vector<Point2f> >& points_vecs,
                           const vector<vector<Point2f> >& flied_points_vecs,
                           const Size patch_size, const Size search_size) {
    assert(points_vecs.size() == image_names.size());

    // flag for the use of flipped points
    bool flip_flag = (flied_points_vecs.size() == points_vecs.size());
    if (flip_flag)
        cout << "PatchContainer : flip_flag is true." << endl;
    else
        cout << "PatchContainer : flip_flag is false." << endl;

    // Set to member variable
    this->base_shape = base_shape;
    this->points_size = base_shape.size();

    // Training window size
    Size window_size = patch_size + search_size;

    // Initialize patches
    this->patches.resize(this->points_size);

    // Start to train
    for (int i = 0; i < points_size; i++) {
        cout << "Training " << i + 1 << "th patch" << endl;

        // Converted images for training
        vector<Mat> training_images;
        training_images.reserve(image_names.size() * (flip_flag ? 2 : 1));

        // original and flipped points
        for (int f = 0; f < (flip_flag ? 2 : 1); f++) {
            // Extract each training region
            for (int j = 0; j < image_names.size(); j++) {
                // Load image
                Mat orl_image = imread(image_names[j], 0);
                if (orl_image.empty()) {
                    cout << "Failed to load : " << image_names[j] << endl;
                    continue;
                }

                // Flip
                vector<Point2f> points;
                if (f == 0) {  // original
                    points = points_vecs[j];
                } else {
                    // Flip image
                    flip(orl_image, orl_image, 1);
                    // Flip points
                    points = flied_points_vecs[j];
                }

                // Affine matrix from base shape (use only rotation ans scale)
                Mat aff_from_base = this->calcAffineFromBase(points);
                // Fix the translation
                this->setAffineRotatedTranslation(aff_from_base, points[i],
                                                  window_size);

                // Warp to base shape (inverse warp)
                Mat warped_image;
                warpAffine(orl_image, warped_image, aff_from_base, window_size,
                           INTER_LINEAR + WARP_INVERSE_MAP);

                // Append
                training_images.push_back(warped_image);
            }
        }
        // Train patch
        this->patches[i].train(training_images, patch_size);
    }
    return;
}

void PatchContainer::calcPeaks(const Mat& src_image,
                               const vector<Point2f>& src_points,
                               vector<Point2f>& dst_points,
                               const Size search_size) {
    assert(this->points_size == src_points.size());

    // Affine matrix from base shape
    Mat aff_from_base = this->calcAffineFromBase(src_points);
    // Affine matrix to base shape
    Mat aff_to_base = this->calcInverseAffine(aff_from_base);

    // Transform input points to base shape
    vector<Point2f> based_points;
    this->applyAffineToPoints(src_points, aff_to_base, based_points);

    // for each points
    for (int i = 0; i < this->points_size; i++) {
        // Window size to get patch response
        Size window_size = this->patches[i].getPatchSize() + search_size;

        // Affine matrix for current point
        Mat each_point_aff = aff_from_base.clone();
        // Fix the translation
        this->setAffineRotatedTranslation(each_point_aff, src_points[i],
                                          window_size);

        // Warp to base shape (inverse warp)
        Mat warped_image;
        warpAffine(src_image, warped_image, each_point_aff, window_size,
                   INTER_LINEAR + WARP_INVERSE_MAP);

        // Get patch response
        Mat response = this->patches[i].calcResponse(warped_image);
        // Find the largest response point
        Point max_point;
        minMaxLoc(response, 0, 0, 0, &max_point);

        // Shift to the response point
        based_points[i].x += max_point.x - 0.5 * search_size.width;
        based_points[i].y += max_point.y - 0.5 * search_size.height;
    }

    // Inverse transform from base shape
    this->applyAffineToPoints(based_points, aff_from_base, dst_points);

    return;
}

void PatchContainer::save(const string& filename) {
    FileStorage cvfs(filename, CV_STORAGE_WRITE);

    write(cvfs, "Base_Shape", this->base_shape);
    write(cvfs, "Points_Size", this->points_size);

    write(cvfs, "Patches_Num", (int)this->patches.size());
    for (int i = 0; i < this->patches.size(); i++) {
        stringstream ss;
        ss << "Patch" << i;
        write(cvfs, ss.str(), this->patches[i].getPatch());
    }
    return;
}

void PatchContainer::load(const string& filename) {
    FileStorage cvfs(filename, CV_STORAGE_READ);
    FileNode node(cvfs.fs, NULL);

    read(node["Base_Shape"], this->base_shape);
    read(node["Points_Size"], this->points_size, 0);

    int patches_size;
    read(node["Patches_Num"], patches_size, 0);
    this->patches.resize(patches_size);

    for (int i = 0; i < this->patches.size(); i++) {
        stringstream ss;
        ss << "Patch" << i;
        Mat patch_mat;
        read(node[ss.str()], patch_mat);
        this->patches[i].setPatch(patch_mat);
    }

    if (this->points_size == 0) {
        cerr << "Invalid Points_Size" << endl;
    }
    if (this->patches.size() == 0) {
        cerr << "Invaild Patch" << endl;
    }

    return;
}

void PatchContainer::visualize() {
    // Get patches and convert to visualize
    vector<Mat> normed_patch(this->patches.size());
    for (int i = 0; i < this->points_size; i++) {
        normed_patch[i] = this->patches[i].getPatch();
        normalize(normed_patch[i], normed_patch[i], 0, 255, CV_MINMAX);
    }

    // Calculate the translation
    Rect bounding_rect = boundingRect(this->base_shape);
    Point shift_amount = -1 * bounding_rect.tl() * 1.4;

    Mat canvas = Mat::zeros(400, 400, CV_8UC1);

    // Start to visualize
    while (true) {
        for (int i = 0; i < this->points_size; i++) {
            normed_patch[i].copyTo(
                canvas(Rect(this->base_shape[i].x + shift_amount.x,
                            this->base_shape[i].y + shift_amount.y,
                            normed_patch[i].cols, normed_patch[i].rows)));

            imshow("Visualized_Patches", canvas);
            if (waitKey(10) == 'q') return;
        }
    }
}

// Calculate affine matrix from base_shape to input points
Mat PatchContainer::calcAffineFromBase(const vector<Point2f>& points) {
    // Assume that base_shape is the origin
    assert(points.size() == this->points_size);

    // Get dst's mean
    float dst_x_mean = 0, dst_y_mean = 0;
    for (int i = 0; i < this->points_size; i++) {
        dst_x_mean += points[i].x;
        dst_y_mean += points[i].y;
    }
    dst_x_mean /= this->points_size;
    dst_y_mean /= this->points_size;

    // Subtract the mean and shift to the origin
    vector<Point2f> dst_points(this->points_size);
    for (int i = 0; i < this->points_size; i++) {
        dst_points[i].x = points[i].x - dst_x_mean;
        dst_points[i].y = points[i].y - dst_y_mean;
    }

    // Calculate scale and rotation
    float a = 0, b = 0, d = 0;
    for (int i = 0; i < this->points_size; i++) {
        float src_0 = this->base_shape[i].x;
        float src_1 = this->base_shape[i].y;
        float dst_0 = dst_points[i].x;
        float dst_1 = dst_points[i].y;
        d += src_0 * src_0 + src_1 * src_1;
        a += src_0 * dst_0 + src_1 * dst_1;
        b += src_0 * dst_1 - src_1 * dst_0;
    }
    a /= d;
    b /= d;

    return (Mat_<float>(2, 3) << a, -b, dst_x_mean, b, a, dst_y_mean);
}

// Update translation element of affine matrix to move the center of point
void PatchContainer::setAffineRotatedTranslation(Mat& affine_mat,
                                                 const Point2f& point,
                                                 const Size& window_size) {
    // Shift to input point (upper left of the search window)
    affine_mat.at<float>(0, 2) =
        point.x - (affine_mat.at<float>(0, 0) * (window_size.width - 1) / 2 +
                   affine_mat.at<float>(0, 1) * (window_size.height - 1) / 2);
    affine_mat.at<float>(1, 2) =
        point.y - (affine_mat.at<float>(1, 0) * (window_size.width - 1) / 2 +
                   affine_mat.at<float>(1, 1) * (window_size.height - 1) / 2);
}

// Calculate inverse of affine matrix
Mat PatchContainer::calcInverseAffine(const Mat& affine_mat) {
    Mat dst_affine(2, 3, CV_32F);
    // Determinant
    float det = affine_mat.at<float>(0, 0) * affine_mat.at<float>(1, 1) -
                affine_mat.at<float>(1, 0) * affine_mat.at<float>(0, 1);
    // Rotation
    dst_affine.at<float>(0, 0) = affine_mat.at<float>(1, 1) / det;
    dst_affine.at<float>(1, 1) = affine_mat.at<float>(0, 0) / det;
    dst_affine.at<float>(0, 1) = -affine_mat.at<float>(0, 1) / det;
    dst_affine.at<float>(1, 0) = -affine_mat.at<float>(1, 0) / det;

    // Rotate translation element and inverse the sign
    Mat rot = dst_affine(Rect(0, 0, 2, 2));
    Mat trans = -1 * rot * affine_mat.col(2);
    // Paste
    trans.copyTo(dst_affine.col(2));

    return dst_affine;
}

// Apply affine matrix
void PatchContainer::applyAffineToPoints(const vector<Point2f>& src_points,
                                         const Mat& aff_mat,
                                         vector<Point2f>& dst_points) {
    dst_points.resize(src_points.size());

    for (int i = 0; i < dst_points.size(); i++) {
        dst_points[i].x = aff_mat.at<float>(0, 0) * src_points[i].x +
                          aff_mat.at<float>(0, 1) * src_points[i].y +
                          aff_mat.at<float>(0, 2);
        dst_points[i].y = aff_mat.at<float>(1, 0) * src_points[i].x +
                          aff_mat.at<float>(1, 1) * src_points[i].y +
                          aff_mat.at<float>(1, 2);
    }
    return;
}
