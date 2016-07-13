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

const string CASCADE_FILE =
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
const string CLM_PATH = "../data/helen_default";

int main(int argc, const char *argv[]) {
    // ====== Initialize ======
    // Initialize connections and symmetry
    vector<int> symmetry;
    vector<Vec2i> connections;
    initHelenConnections(connections);
    initHelenSymmetry(symmetry);
    // CLM
    Clm clm(CLM_PATH, CASCADE_FILE);
    // Fps
    FpsCounter fps;

    // ====== Tracking ======
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "No Webcam." << endl;
        return 1;
    }
    Mat orl_image, image;
    vector<Point2f> points;

    bool init_flag = false;
    bool output_fps_flag = true;

    while (true) {
        cap >> orl_image;
        flip(orl_image, orl_image, 1);
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
