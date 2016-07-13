#include "detector.h"

using namespace cv;
using namespace std;

const float Detector::SMALL_IMAGE_SCALE = 0.2;

Detector::Detector() {}

Detector::Detector(const string& cascade_file) {
    this->initCascade(cascade_file);
}

void Detector::initCascade(const string& cascade_file) {
    this->cascade.load(cascade_file);
}

void Detector::train(const vector<Point2f>& shape,
                     const vector<string>& image_names,
                     const vector<vector<Point2f> >& points_vecs,
                     const vector<vector<Point2f> >& flied_points_vecs,
                     const float cascade_scale,
                     const int cascade_min_neighbours,
                     const Size cascade_min_size, const float bounding_per) {
    assert(image_names.size() == points_vecs.size());

    if (this->cascade.empty()) {
        cerr << "Error: cascade is not loaded" << endl;
        return;
    }

    // flag for the use of flipped points
    bool flip_flag = (flied_points_vecs.size() == points_vecs.size());
    if (flip_flag)
        cout << "Detector : flip_flag is true." << endl;
    else
        cout << "Detector : flip_flag is false." << endl;

    // Set to member variable
    this->base_shape = shape;

    // Face coordinates and scales
    vector<float> offset_x, offset_y, offset_scale;
    offset_x.reserve(image_names.size() * (flip_flag ? 2 : 1));
    offset_y.reserve(image_names.size() * (flip_flag ? 2 : 1));
    offset_scale.reserve(image_names.size() * (flip_flag ? 2 : 1));

    // original and flipped
    for (int f = 0; f < (flip_flag ? 2 : 1); f++) {
        // for each training data
        for (int i = 0; i < image_names.size(); i++) {
            // Load image
            Mat orl_image = imread(image_names[i], 0);
            if (orl_image.empty()) {
                cout << "Failed to load : " << image_names[i] << endl;
                continue;
            }

            // Flip
            vector<Point2f> points;
            if (f == 0) {  // original
                points = points_vecs[i];
            } else {
                // flip image
                flip(orl_image, orl_image, 1);
                // flip points
                points = flied_points_vecs[i];
            }

            // Equalize histogram
            Mat normed_image;
            equalizeHist(orl_image, normed_image);
            // Detect the largest face
            vector<Rect> face_rects;
            this->cascade.detectMultiScale(
                normed_image, face_rects, cascade_scale, cascade_min_neighbours,
                0 | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
                cascade_min_size);

            // Skip when detection is failed
            if (face_rects.size() == 0) continue;

            // == Debug visualize ==
            // Mat canvas = orl_image.clone();
            // for (int n = 0; n < points.size(); n++) {
            //     circle(canvas, points[n], 1, Scalar(0,255,0), 2, CV_AA);
            // }
            // rectangle(canvas, face_rects[0], Scalar(255));
            // imshow("Detector Training", canvas);
            // waitKey(10);

            // Check if the shape is valid
            if (this->isBoundingEnough(points, face_rects[i], bounding_per)) {
                // Get shape center
                Point2f center = this->calcMassCenter(points);
                float width = face_rects[0].width;

                // Get rectangle center
                Point2f rect_center = face_rects[0].tl() + face_rects[0].br();
                rect_center.x *= 0.5;
                rect_center.y *= 0.5;

                // Append results
                offset_x.push_back((center.x - rect_center.x) / width);
                offset_y.push_back((center.y - rect_center.y) / width);
                offset_scale.push_back(this->calcScaleForBase(points) / width);
            }
        }
    }

    // Sort the results
    Mat x, y, scale;
    cv::sort(Mat(offset_x), x, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
    cv::sort(Mat(offset_y), y, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
    cv::sort(Mat(offset_scale), scale,
             CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);

    // Save the median value
    this->offsets = Vec3f(x.at<float>(x.rows / 2), y.at<float>(y.rows / 2),
                          scale.at<float>(scale.rows / 2));
}

void Detector::detect(const Mat& image, vector<Point2f>& dst_points,
                      const float cascade_scale,
                      const int cascade_min_neighbours,
                      const Size cascade_min_size) {
    if (this->cascade.empty()) {
        cerr << "Error: cascade is not loaded" << endl;
        return;
    }

    dst_points.clear();

    // Convert to gray image
    Mat gray_image;
    if (image.channels() == 1)
        gray_image = image;
    else
        cvtColor(image, gray_image, CV_RGB2GRAY);

    // Equalize histogram
    equalizeHist(gray_image, gray_image);
    // Detect largest face
    vector<Rect> face_rects;
    this->cascade.detectMultiScale(
        gray_image, face_rects, cascade_scale, cascade_min_neighbours,
        0 | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
        cascade_min_size);

    // exit when failed to detect
    if (face_rects.size() == 0) return;

    // Shift base points
    int points_size = this->base_shape.size();
    dst_points.resize(points_size);

    // Calculate the difference
    Vec3f cur_offsets = this->offsets * face_rects[0].width;
    float shift_x =
        face_rects[0].x + 0.5 * face_rects[0].width + cur_offsets[0];
    float shift_y =
        face_rects[0].y + 0.5 * face_rects[0].height + cur_offsets[1];
    // Apply for each points
    for (int i = 0; i < points_size; i++) {
        dst_points[i].x = cur_offsets[2] * this->base_shape[i].x + shift_x;
        dst_points[i].y = cur_offsets[2] * this->base_shape[i].y + shift_y;
    }

    // Prepare for re-detection rectangle and image
    this->face_rect = face_rects[0];
    resize(image(face_rect), this->face_small, Size(), SMALL_IMAGE_SCALE,
           SMALL_IMAGE_SCALE, INTER_AREA);

    return;
}

void Detector::redetect(const Mat& image, vector<Point2f>& points) {
    // Check the first detection
    if (this->face_small.empty() || this->face_rect.x == 0 ||
        this->face_rect.y == 0) {
        // first detection
        this->detect(image, points);
        return;
    }

    // Template matching
    Mat small_image;
    Mat response;
    resize(image, small_image, Size(), SMALL_IMAGE_SCALE, SMALL_IMAGE_SCALE,
           INTER_AREA);
    matchTemplate(small_image, this->face_small, response, CV_TM_CCOEFF_NORMED);
    // Make the sum 1
    normalize(response, response, 0, 1, NORM_MINMAX);
    response /= sum(response)[0];
    // Get position where response is the largest
    Point max_point;
    minMaxLoc(response, 0, 0, 0, &max_point);

    // When region is out of image, exit
    if (image.cols <= max_point.x / SMALL_IMAGE_SCALE + this->face_rect.width ||
        image.rows <= max_point.y / SMALL_IMAGE_SCALE + this->face_rect.height)
        return;

    float shift_x = max_point.x / SMALL_IMAGE_SCALE - this->face_rect.x;
    float shift_y = max_point.y / SMALL_IMAGE_SCALE - this->face_rect.y;
    // Apply shift
    for (int i = 0; i < points.size(); i++) {
        points[i].x += shift_x;
        points[i].y += shift_y;
    }

    // Prepare for next re-detection
    this->face_rect.x = max_point.x / SMALL_IMAGE_SCALE;
    this->face_rect.y = max_point.y / SMALL_IMAGE_SCALE;
    resize(image(face_rect), this->face_small, Size(), SMALL_IMAGE_SCALE,
           SMALL_IMAGE_SCALE, INTER_AREA);

    // imshow("face_small", this->face_small);
}

void Detector::save(const string& filename) {
    FileStorage cvfs(filename, CV_STORAGE_WRITE);
    write(cvfs, "Base_Shape", this->base_shape);
    write(cvfs, "Offsets", this->offsets);
}

void Detector::load(const string& filename) {
    FileStorage cvfs(filename, CV_STORAGE_READ);
    FileNode node(cvfs.fs, NULL);
    read(node["Base_Shape"], this->base_shape);
    read(node["Offsets"], this->offsets, Vec3f());
}

void Detector::visualize() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "No Webcam." << endl;
        return;
    }

    Mat image;
    for (;;) {
        cap >> image;

        // detect
        vector<Point2f> points;
        this->detect(image, points);

        // draw
        for (int i = 0; i < points.size(); i++) {
            circle(image, points[i], 1, Scalar(0, 255, 0), -1, 8, 0);
        }

        imshow("detect", image);
        if (waitKey(10) == 'q') break;
    }
}

void Detector::visualize(const vector<string>& image_names) {
    for (int i = 0; i < image_names.size(); i++) {
        Mat image = imread(image_names[i], 0);

        // detect
        vector<Point2f> points;
        this->detect(image, points);
        if (points.size() == 0) {
            cerr << "No faces" << endl;
            continue;
        }

        // draw
        for (int i = 0; i < points.size(); i++) {
            circle(image, points[i], 1, Scalar(200), -1, 8, 0);
        }

        imshow("detect", image);
        if (waitKey(0) == 'q') break;
    }
}

// private
bool Detector::isBoundingEnough(const vector<Point2f>& points, const Rect& rect,
                                const float percent) {
    int inside_points_num = 0;
    // check for each points and count
    for (int i = 0; i < points.size(); i++) {
        if (rect.contains(points[i])) inside_points_num++;
    }

    // check the rate
    if ((float)inside_points_num / (float)points.size())
        return true;
    else
        return false;
}

Point2f Detector::calcMassCenter(const vector<Point2f>& points) {
    Point2f mean_point(0, 0);
    for (int i = 0; i < points.size(); i++) {
        mean_point += points[i];
    }
    mean_point.x /= points.size();
    mean_point.y /= points.size();

    return mean_point;
}

float Detector::calcScaleForBase(const vector<Point2f>& points) {
    Point2f center = this->calcMassCenter(points);
    float scale = 0, base = 0;
    for (int i = 0; i < points.size(); i++) {
        scale += (points[i] - center).dot(this->base_shape[i]);
        base += this->base_shape[i].dot(this->base_shape[i]);
    }
    return scale / base;
}
