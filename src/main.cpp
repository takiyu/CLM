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

int main(int argc, const char *argv[]) {
    // ===== Arguments =====
    string cascade_file =
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
    string clm_path = "../data/helen_default";
    string dataset_mode = "helen";
    string image_path = "";
    for (int i = 1; i < argc; i++) {
        string arg(argv[i]);
        if (arg == "--cascade") {
            cascade_file = string(argv[++i]);
        } else if (arg == "--clm") {
            clm_path = string(argv[++i]);
        } else if (arg == "--datamode") {
            dataset_mode = string(argv[++i]);
        } else if (arg == "--image") {
            image_path = string(argv[++i]);
        } else if (arg == "-h") {
            cout << "Arguments" << endl;
            cout << " --cascade <path>" << endl;
            cout << " --clm <path>" << endl;
            cout << " --datamode [helen/muct]" << endl;
            cout << " --image <path>" << endl;
            return 0;
        }
    }
    cout << " >> Cascade file: " << cascade_file << endl;
    cout << " >> CLM path: " << clm_path << endl;
    cout << " >> Data mode: " << dataset_mode << endl;
    cout << " >> Image path: " << image_path << endl;
    cout << endl;

    // ====== Initialize ======
    // Initialize connections and symmetry
    vector<int> symmetry;
    vector<Vec2i> connections;
    if (dataset_mode == "helen") {
        initHelenConnections(connections);
        initHelenSymmetry(symmetry);
    } else if (dataset_mode == "muct") {
        initMuctConnections(connections);
        initMuctSymmetry(symmetry);
    } else {
        cout << "Invalid data mode" << endl;
        return 1;
    }
    // CLM
    Clm clm(clm_path, cascade_file);
    // Fps
    FpsCounter fps;

    // ====== Tracking ======
    VideoCapture cap;
    if (image_path.empty()) {
        cap.open(0);
        if (!cap.isOpened()) {
            cerr << "No Webcam." << endl;
            return 1;
        }
    }
    Mat orl_image, image;
    vector<Point2f> points;

    bool init_flag = false;
    bool output_fps_flag = true;

    while (true) {
        if (image_path.empty()) {
            cap >> orl_image;
            flip(orl_image, orl_image, 1);
        } else {
            orl_image = imread(image_path);
        }
        image = orl_image;

        // track
        if (clm.track(image, points, init_flag, true)) {
            init_flag = false;

            // draw result
            Shape::drawPoints(orl_image, points, Scalar(0, 255, 0), 2,
                              connections);
            // Shape::drawPointsWithIdx(orl_image, points, Scalar(0, 255, 0), 2,
            //                          connections);
        }

        fps.updateFrame();
        if (output_fps_flag) cout << "fps:" << fps.getFps() << endl;

        imshow("image", orl_image);
        char key = waitKey(5);

        if (key == 'q')
            break;
        else if (key == 'd')
            init_flag = true;
        else if (key == 'f')
            output_fps_flag = !output_fps_flag;
    }

    return 0;
}
