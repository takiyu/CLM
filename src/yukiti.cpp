#include "clm/shape.h"
#include "clm/patch.h"
#include "clm/detector.h"
#include "clm/clm.h"
#include "appearance/appearance.h"
#include "io/face_data.h"
#include "io/fps.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <dirent.h>

using namespace cv;
using namespace std;

Mat imreadWithErr(const string& filename, const int flag){
	if(filename.compare("") == 0){ cerr << "imreadWithErr : filenameが空文字 " << endl; return Mat(); }
	Mat image = imread(filename, flag);
	if(image.empty()){ cerr << "imreadWithErr : 読み取りに失敗(" << filename << ")" << endl; }
	return image;
}

int main(int argc, const char *argv[]) {
// 	const string CASCADE_FILE = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt_tree.xml";
	const string CASCADE_FILE = "/usr/share/opencv/lbpcascades/lbpcascade_frontalcatface.xml";

	vector<int> symmetry;
	vector<Vec2i> connections;

	//connectionsの初期化
	initHelenConnections(connections);
	//symmetryの初期化
	initHelenSymmetry(symmetry);

	//=====Init Appearance=====
	//マスク顔画像読み込み
	Mat base_app_image = imreadWithErr("../data/yukiti_data/dst_yukiti_face.jpeg", 1);
	//画像に対応する点群を読み込み
	vector<Point2f> base_app_shape, tmp_readed_shape;
	FileStorage cvfs("../data/yukiti_data/dst_yukiti_face.ptdata", CV_STORAGE_READ);
	FileNode node(cvfs.fs, NULL);
	read(node["Points"], tmp_readed_shape);
	//数を減らす
	reduceHelenPoints(tmp_readed_shape, base_app_shape);
	//平面分割
	App app(base_app_shape);


	//======初期化======
	Clm clm("../data/helen_default", CASCADE_FILE);
	FpsCounter fps;


	//======Bill Image=====
	//CANVAS_RECTの高さと座標に合わせて描画される(幅を狭めることで高速化可能)
	Rect CANVAS_RECT(0,0, 170,170);
	//背景画像
	Mat orl_bill_image = Mat::zeros(170,170,CV_8UC3);
	Mat bill_image = orl_bill_image;//初回権出失敗対策
	Mat canvas;

	//======Tracking======
	VideoCapture cap(0);
	if(!cap.isOpened()){ cerr << "No Webcam." << endl; return 1; }
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

	for(;;){
		cap >> orl_image;
		flip(orl_image, orl_image, 1);
// 		GaussianBlur(orl_image, image, Size(3,3), 5);
		image = orl_image;

		//追跡
		if(clm.track(image, points, init_flag, true)){
			//成功時
			init_flag = false;

			//お札の画像を初期化
			bill_image = orl_bill_image.clone();
			canvas = bill_image(CANVAS_RECT);

			//点群をwarp用に一部削除
			reduceHelenPoints(points, app_points);
			//点群の位置をcanvasに合わせる
			Rect app_bounding = boundingRect(app_points);
			Shape::shiftPoints(app_points, (app_bounding.br()+app_bounding.tl()) * -0.5);
			Shape::resizePoints(app_points, CANVAS_RECT.height/(float)app_bounding.height);
			Shape::shiftPoints(app_points, Point2f(CANVAS_RECT.size()) * 0.5);

			//warp (時間がかかる場合は範囲を狭めるか、GPU支援を利用)
			app.warp(base_app_image, canvas, base_app_shape, app_points);

			//追跡結果を描画
			if(draw_src_points_flag){
				Shape::drawPoints(orl_image, points, Scalar(0,255,0), 2, connections);
// 				Shape::drawPointsWithIdx(orl_image, points, Scalar(0,255,0), 2, connections);
			}
			if(draw_canvas_rect_flag){
				rectangle(bill_image, CANVAS_RECT, Scalar(0,255,0));
			}
			if(draw_bill_points_flag){
				Shape::drawPoints(canvas, app_points, Scalar(0,255,0), 2);
			}

			resize(bill_image, bill_image, Size(), 1.5, 1.5, INTER_LINEAR);
		}

		current_fps = fps.getFps();
		if(current_fps == 0) current_fps = 1;
		fps.updateFrame();
		if(output_fps_flag) cout << "fps:" << current_fps << endl;

		std::cout << "auto reset:" << auto_reset_flag << std::endl;

		imshow("bill", bill_image);
		imshow("image", orl_image);
		char key = waitKey(5);

		if(key == 'q') break; //終了
		else if(key == 'd') init_flag = true; //再検出
		//描画フラグ
		else if(key == 'm') draw_src_points_flag = !draw_src_points_flag;
		else if(key == 'n') draw_canvas_rect_flag = !draw_canvas_rect_flag;
		else if(key == 'b') draw_bill_points_flag = !draw_bill_points_flag;
		//fps表示フラグ
		else if(key == 'f') output_fps_flag = !output_fps_flag;
		//自動リセットフラグ
		else if(key == 'r') auto_reset_flag = !auto_reset_flag;

		//自動再検出
		if(auto_reset_flag && ++frame_count >= current_fps * 5){
			init_flag = true;
			frame_count = 0;
		}
	}

	return 0;
}
