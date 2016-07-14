#include "clm/clm.h"
#include "clm/detector.h"
#include "clm/patch.h"
#include "clm/shape.h"
#include "io/face_data.h"

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
    string out_dir = "";
    string cascade_file =
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
    string muct_image_dir = "./muct/jpg/";
    string muct_lm_file = "./muct/muct-landmarks/muct76-opencv.csv";
    for (int i = 1; i < argc; i++) {
        string arg(argv[i]);
        if (arg == "--out") {
            out_dir = string(argv[++i]);
        } else if (arg == "--cascade") {
            cascade_file = string(argv[++i]);
        } else if (arg == "--muct_image_dir") {
            muct_image_dir = string(argv[++i]);
        } else if (arg == "--muct_lm_file") {
            muct_lm_file = string(argv[++i]);
        } else if (arg == "-h") {
            cout << "Arguments" << endl;
            cout << " --out <path>" << endl;
            cout << " --cascade <path>" << endl;
            cout << " --muct_image_dir <path>" << endl;
            cout << " --muct_lm_file <path>" << endl;
            return 0;
        }
    }
    cout << " >> Output directory: " << out_dir << endl;
    cout << " >> Cascade file: " << cascade_file << endl;
    cout << " >> MUCT image dir: " << muct_image_dir << endl;
    cout << " >> MUCT landmark: " << muct_lm_file << endl;
    cout << endl;

    if (out_dir.empty()) {
        cout << "No output directory argument" << endl;
        return 1;
    }

    vector<string> image_names;
    vector<vector<Point2f> > points_vecs;
    vector<int> symmetry;
    vector<Vec2i> connections;

    //====== MUCT Data ====== begin
    readMUCTLandMarksFile(muct_lm_file, muct_image_dir, image_names,
                          points_vecs);
    removeIncompleteShape(points_vecs, image_names);
    initMuctConnections(connections);
    initMuctSymmetry(symmetry);
    //====== MUCT Data ====== end

    //====== Helen Data (Broken now) (TODO: Fix) ====== begin
    // cout << "Load helen dataset" << endl;
    // readHelenFiles(HELEN_IMAGE_DIR, HELEN_POINT_DIR, image_names,
    // points_vecs);
    // initHelenConnections(connections);
    // initHelenSymmetry(symmetry);
    //====== Helen Data====== end

    cout << "Start to train" << endl;
    Clm::train(image_names, points_vecs, cascade_file, symmetry, connections,
               out_dir);
    return 0;
}
