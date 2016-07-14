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

void insertCornerPoints(vector<Point2f>& points, const Mat& img) {
    points.push_back(Point2f(0, 0));
    points.push_back(Point2f(img.cols, 0));
    points.push_back(Point2f(0, img.rows));
    points.push_back(Point2f(img.cols, img.rows));
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
    vector<Point2f> bill_shape, tmp_readed_shape;
    FileStorage cvfs("../data/yukiti_data/bill_yukiti.ptdata", CV_STORAGE_READ);
    FileNode node(cvfs.fs, NULL);
    read(node["Points"], tmp_readed_shape);
    // Reduce the number of points
    reduceHelenPoints(tmp_readed_shape, bill_shape);

    // Create editing region
    const int BILL_RECT_OFFSET = 50;
    Rect BILL_RECT = boundingRect(bill_shape);
    Rect ORL_BILL_RECT = BILL_RECT;
    BILL_RECT.x -= BILL_RECT_OFFSET;
    BILL_RECT.y -= BILL_RECT_OFFSET;
    BILL_RECT.width += BILL_RECT_OFFSET * 2;
    BILL_RECT.height += BILL_RECT_OFFSET * 2;

    // Apply region
    Mat orl_bill_image = bill_image_full(BILL_RECT).clone();
    for (int i = 0; i < bill_shape.size(); i++) {
        bill_shape[i] -= Point2f(BILL_RECT.x, BILL_RECT.y);
    }

    //===== Initialize Appearance =====
    // Insert image corner points
    vector<Point2f> corner_bill_shape = bill_shape;
    insertCornerPoints(corner_bill_shape, orl_bill_image);
    // Appearance
    App app(corner_bill_shape);

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
    Mat orl_image, image;
    vector<Point2f> points;
    vector<Point2f> app_points;

    bool init_flag = false;
    bool draw_src_points_flag = true;
    bool draw_bill_points_flag = false;
    bool draw_canvas_rect_flag = false;
    bool output_fps_flag = true;
    bool auto_reset_flag = true;

    int frame_count = 0;
    int current_fps = 15;

    while (true) {
        cap >> orl_image;
        flip(orl_image, orl_image, 1);
        image = orl_image;

        // Track
        if (clm.track(image, points, init_flag, true)) {
            init_flag = false;

            // Setup current points to warp
            reduceHelenPoints(points, app_points);
            // Fit points to bill region
            Rect app_bounding = boundingRect(app_points);
            Shape::shiftPoints(app_points,
                               (app_bounding.br() + app_bounding.tl()) * -0.5);
            Shape::resizePoints(
                app_points, ORL_BILL_RECT.height / (float)app_bounding.height);
            Shape::shiftPoints(app_points,
                               Point2f(ORL_BILL_RECT.size()) * 0.5 +
                                   Point2f(BILL_RECT_OFFSET, BILL_RECT_OFFSET));
            // Insert image corner points
            insertCornerPoints(app_points, orl_bill_image);

            // Image region to write over
            Mat bill_image = bill_image_full(BILL_RECT);

            // Warp
            app.warp(orl_bill_image, bill_image, corner_bill_shape, app_points);

            // Draw tracking result
            if (draw_src_points_flag) {
                Shape::drawPoints(orl_image, points, Scalar(0, 255, 0), 2,
                                  connections);
            }
            if (draw_canvas_rect_flag) {
                rectangle(bill_image, BILL_RECT, Scalar(0, 255, 0));
            }
            if (draw_bill_points_flag) {
                Shape::drawPoints(bill_image, app_points, Scalar(0, 255, 0), 2);
            }

            resize(bill_image, bill_image, Size(), 1.5, 1.5, INTER_LINEAR);
        }

        current_fps = fps.getFps();
        if (current_fps == 0) current_fps = 1;
        fps.updateFrame();
        if (output_fps_flag) cout << "fps:" << current_fps << endl;

        std::cout << "auto reset:" << auto_reset_flag << std::endl;

        imshow("bill", bill_image_full);
        imshow("image", orl_image);
        char key = waitKey(5);

        // Exit
        if (key == 'q') break;
        // Re-detection
        else if (key == 'd')
            init_flag = true;
        // Drawing flags
        else if (key == 'm')
            draw_src_points_flag = !draw_src_points_flag;
        else if (key == 'n')
            draw_canvas_rect_flag = !draw_canvas_rect_flag;
        else if (key == 'b')
            draw_bill_points_flag = !draw_bill_points_flag;
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
