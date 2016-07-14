#include "appearance/appearance.h"
#include "clm/clm.h"
#include "clm/detector.h"
#include "clm/patch.h"
#include "clm/shape.h"
#include "io/face_data.h"
#include "io/fps.h"

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

void fillMask(const vector<Point2f>& points, Mat& mask) {
    assert(mask.type() == CV_8UC1);
    // vector<Point2f> -> vector<Point>
    vector<Point> tmp_points(points.size());
    for (int i = 0; i < points.size(); i++) {
        const Point2f& pt = points[i];
        tmp_points[i] = Point(pt.x, pt.y);
    }

    // Create mask
    vector<Point> hull;
    convexHull(tmp_points, hull);
    fillConvexPoly(mask, hull, Scalar(255));
}

int main(int argc, const char* argv[]) {
    // ===== Arguments =====
    string cascade_file =
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
    string clm_path = "../data/helen_default";
    for (int i = 1; i < argc; i++) {
        string arg(argv[i]);
        if (arg == "--cascade") {
            cascade_file = string(argv[++i]);
        } else if (arg == "--clm") {
            clm_path = string(argv[++i]);
        } else if (arg == "-h") {
            cout << "Arguments" << endl;
            cout << " --cascade <path>" << endl;
            cout << " --clm <path>" << endl;
            return 0;
        }
    }
    cout << " >> Cascade file: " << cascade_file << endl;
    cout << " >> CLM path: " << clm_path << endl;
    cout << endl;

    //====== Initialize Bill Data =====
    // Load image
    Mat bill_image_full = imread("../data/yukiti_data/bill_yukiti.jpeg");
    // Load shape
    vector<Point2f> bill_shape;
    FileStorage cvfs("../data/yukiti_data/bill_yukiti.ptdata", CV_STORAGE_READ);
    FileNode node(cvfs.fs, NULL);
    read(node["Points"], bill_shape);

    // Create editing region
    Rect BILL_RECT = boundingRect(bill_shape);

    // Apply region
    Mat orl_bill_image = bill_image_full(BILL_RECT).clone();
    for (int i = 0; i < bill_shape.size(); i++) {
        bill_shape[i] -= Point2f(BILL_RECT.x, BILL_RECT.y);
    }

    //===== Initialize Appearance =====
    // Appearance
    Appearance app(bill_shape);

    //====== Initialize ======
    // Initialize connections and symmetry
    vector<int> symmetry;
    vector<Vec2i> connections;
    initHelenConnections(connections);
    initHelenSymmetry(symmetry);
    // CLM
    Clm clm(clm_path, cascade_file);
    // Fps
    FpsCounter fps;

    //====== Tracking ======
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "No Webcam." << endl;
        return 1;
    }
    Mat image;
    vector<Point2f> points;

    bool init_flag = false;
    bool draw_src_points_flag = false;
    bool output_fps_flag = true;
    bool auto_reset_flag = true;

    int frame_count = 0;
    int current_fps = 15;

    while (true) {
        cap >> image;
        flip(image, image, 1);

        // Track
        if (clm.track(image, points, init_flag, true)) {
            init_flag = false;

            // Image region to write over
            Rect bbox = boundingRect(points);
            // check bbox position
            bool valid_bbox = (0 <= bbox.x && 0 <= bbox.y &&
                               bbox.x + bbox.width < image.cols &&
                               bbox.y + bbox.height < image.rows);
            if (valid_bbox) {
                // ROI for speed up
                Mat warped_image = image(bbox).clone();
                for (int i = 0; i < points.size(); i++) {
                    points[i] -= Point2f(bbox.x, bbox.y);
                }

                // Warp
                app.warp(orl_bill_image, warped_image, bill_shape, points);

                // Create mask
                Mat mask = Mat::zeros(bbox.height, bbox.width, CV_8UC1);
                fillMask(points, mask);

                // Correct warped_image's shading
                int blur_amount = int(bbox.width * 0.4f) | 1;  // odd
                Size blur_kernel(blur_amount, blur_amount);
                Mat image_blur, bill_blur;
                GaussianBlur(image(bbox), image_blur, blur_kernel, 0);
                GaussianBlur(warped_image, bill_blur, blur_kernel, 0);
                warped_image.convertTo(warped_image, CV_32FC3);
                image_blur.convertTo(image_blur, CV_32FC3);
                bill_blur.convertTo(bill_blur, CV_32FC3);
                cv::multiply(warped_image, image_blur, warped_image);
                cv::divide(warped_image, bill_blur, warped_image);
                warped_image.convertTo(warped_image, CV_8UC3);

                // Over write
                warped_image.copyTo(image(bbox), mask);

                // Draw tracking result
                if (draw_src_points_flag) {
                    Shape::drawPoints(image, points, Scalar(0, 255, 0), 2,
                                      connections);
                }
            }
        }

        current_fps = fps.getFps();
        if (current_fps == 0) current_fps = 1;
        fps.updateFrame();
        if (output_fps_flag) cout << "fps:" << current_fps << endl;

        std::cout << "auto reset:" << auto_reset_flag << std::endl;

        imshow("image", image);
        char key = waitKey(5);

        // Exit
        if (key == 'q') break;
        // Re-detection
        else if (key == 'd')
            init_flag = true;
        // Drawing flags
        else if (key == 'm')
            draw_src_points_flag = !draw_src_points_flag;
        // Fps flag
        else if (key == 'f')
            output_fps_flag = !output_fps_flag;
        // Auto re-detect flag
        else if (key == 'r')
            auto_reset_flag = !auto_reset_flag;

        // For re-detection
        if (auto_reset_flag && ++frame_count >= current_fps * 5) {
            init_flag = true;
            frame_count = 0;
        }
    }

    return 0;
}
