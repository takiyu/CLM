#include "clm/shape.h"
#include "clm/patch.h"
#include "clm/detector.h"
#include "clm/clm.h"
#include "io/face_data.h"

#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <dirent.h>

using namespace cv;
using namespace std;

int main(int argc, const char *argv[]) {
	if(argc != 2){
		cout << "===使い方===" << endl;
		cout << "train.out <出力先ディレクトリ>" << endl;
		cout << "その他詳細はソースコード参照" << endl;

		return 0;
	}
	const string CASCADE_FILE = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
	const string OUT_DIR(argv[1]);

	vector<string> image_names;
	vector<vector<Point2f> > points_vecs;
	vector<int> symmetry;
	vector<Vec2i> connections;
	
// 	//======MUCT LandMarks を読み取り====== begin
// 	const string IMAGE_DIR = "/media/sf_DocumentFolder/face/MUCT/jpg/";
// 	const string LM_FILE = IMAGE_DIR+"muct-landmarks/muct76-opencv.csv";
// 	//read
// 	readMUCTLandMarksFile(LM_FILE, IMAGE_DIR, image_names, points_vecs);
// 	//(0,0)を含む形状を削除
// 	removeIncompleteShape(points_vecs, image_names);
// 	//connectionsの初期化
// 	initMuctConnections(connections);
// 	//symmetryの初期化
// 	initMuctSymmetry(symmetry);
// 	//======MUCT LandMarks を読み取り====== end

	//======Helen学習データを読み取り====== begin
	const string HELEN_IMAGE_DIR = "/mnt/storage/OnlineStorage/GoogleDrive/Documents/face/helen/trainset/Image";
	const string HELEN_POINT_DIR = "/mnt/storage/OnlineStorage/GoogleDrive/Documents/face/helen/trainset/Points";
 	//read
	readHelenFiles(HELEN_IMAGE_DIR, HELEN_POINT_DIR, image_names, points_vecs);
	//connectionsの初期化
	initHelenConnections(connections);
	//symmetryの初期化
	initHelenSymmetry(symmetry);
	//======Helen学習データを読み取り====== end

	Clm::train(image_names, points_vecs, CASCADE_FILE, symmetry, connections, OUT_DIR);
	return 0;
}
