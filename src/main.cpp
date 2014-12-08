#include "clm/shape.h"
#include "clm/patch.h"
#include "clm/detector.h"
#include "clm/clm.h"
#include "io/face_data.h"
#include "io/fps.h"

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
	const string CASCADE_FILE = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";

	vector<int> symmetry;
	vector<Vec2i> connections;

	//connectionsの初期化
	initHelenConnections(connections);
	//symmetryの初期化
	initHelenSymmetry(symmetry);

	//======初期化======
	Clm clm("../data/helen_default", CASCADE_FILE);
// 	Clm clm("../data/helen_default/shape.data","../data/helen_default/patch.data","../data/helen_default/detector.data", CASCADE_FILE);
	FpsCounter fps;

	//======Tracking======
	VideoCapture cap(0);
	if(!cap.isOpened()){ cerr << "No Webcam." << endl; return 1; }
	Mat orl_image, image;
	vector<Point2f> points;
	vector<Point2f> app_points;

	bool init_flag = false;
	bool output_fps_flag = true;

	for(;;){
		cap >> orl_image;
		flip(orl_image, orl_image, 1);
// 		GaussianBlur(orl_image, image, Size(3,3), 5);
// 		cvtColor(orl_image, orl_image, CV_BGR2GRAY);
		image = orl_image;

		//追跡
		if(clm.track(image, points, init_flag, true)){
			//成功時
			init_flag = false;

			//追跡結果を描画
			Shape::drawPoints(orl_image, points, Scalar(0,255,0), 2, connections);
// 			Shape::drawPointsWithIdx(orl_image, points, Scalar(0,255,0), 2, connections);
		}

		fps.updateFrame();
		if(output_fps_flag) cout << "fps:" << fps.getFps() << endl;

		imshow("image", orl_image);
		char key = waitKey(5);

		if(key == 'q') break; //終了
		else if(key == 'd') init_flag = true; //再検出
		//fps表示フラグ
		else if(key == 'f') output_fps_flag = !output_fps_flag;
	}

	return 0;
}
