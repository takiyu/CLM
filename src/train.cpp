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

const string CASCADE_FILE =
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";

// MUCT
// const string MUCT_IMAGE_DIR = "/media/sf_DocumentFolder/face/MUCT/jpg/";
// const string MUCT_LM_FILE = MUCT_IMAGE_DIR +
// "muct-landmarks/muct76-opencv.csv";

// Helen
const string HELEN_IMAGE_DIR =
    "/mnt/storage/OnlineStorage/GoogleDrive/Documents/face/helen/trainset/"
    "Image";
const string HELEN_POINT_DIR =
    "/mnt/storage/OnlineStorage/GoogleDrive/Documents/face/helen/trainset/"
    "Points";

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        cout << "=== Usage ===" << endl;
        cout << "train.out <out dir>" << endl;
        return 0;
    }
    const string OUT_DIR(argv[1]);

    vector<string> image_names;
    vector<vector<Point2f> > points_vecs;
    vector<int> symmetry;
    vector<Vec2i> connections;

    //====== MUCT Data ====== begin
    // readMUCTLandMarksFile(MUCT_LM_FILE, MUCT_IMAGE_DIR, image_names,
    // points_vecs);
    // removeIncompleteShape(points_vecs, image_names);
    // initMuctConnections(connections);
    // initMuctSymmetry(symmetry);
    //====== MUCT Data ====== end

    //====== Helen Data ====== begin
    readHelenFiles(HELEN_IMAGE_DIR, HELEN_POINT_DIR, image_names, points_vecs);
    initHelenConnections(connections);
    initHelenSymmetry(symmetry);
    //====== Helen Data====== end

    Clm::train(image_names, points_vecs, CASCADE_FILE, symmetry, connections,
               OUT_DIR);
    return 0;
}
