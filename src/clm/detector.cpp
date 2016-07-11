#include "detector.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

//redetectに使用する縮小サイズ
const float Detector::SMALL_IMAGE_SCALE = 0.2;

//public
//コンストラクタ
Detector::Detector(){
}
Detector::Detector(const string& cascade_file){
	this->initCascade(cascade_file);
}
//カスケードの初期化
void Detector::initCascade(const string& cascade_file){
	this->cascade.load(cascade_file);
}
//学習
void Detector::train(const vector<Point2f>& shape, const vector<string>& image_names, const vector<vector<Point2f> >& points_vecs, const vector<vector<Point2f> >& flied_points_vecs, const float cascade_scale, const int cascade_min_neighbours, const Size cascade_min_size, const float bounding_per){
	assert(image_names.size() == points_vecs.size());

	if(this->cascade.empty()){
		cerr << "cascadeが未初期化" << endl;
		return;
	}

	//反転データを使用するかのフラグ
	bool flip_flag = (flied_points_vecs.size() == points_vecs.size());
	if(flip_flag) cout << "Detector : flip_flag is true." << endl;
	else cout << "Detector : flip_flag is false." << endl;

	//メンバ変数を準備
	this->base_shape = shape;

	//検出された顔の座標、スケールを一時保存する変数
	vector<float> offset_x, offset_y, offset_scale;
	offset_x.reserve(image_names.size() * (flip_flag?2:1));
	offset_y.reserve(image_names.size() * (flip_flag?2:1));
	offset_scale.reserve(image_names.size() * (flip_flag?2:1));


	//オリジナルと反転分をそれぞれ計算
	for(int f = 0; f < (flip_flag ? 2 : 1); f++){
		//各学習データについて
		for(int i = 0; i < image_names.size(); i++){

			//画像読み取り
			Mat orl_image = imread(image_names[i], 0);
			if(orl_image.empty()){
				cout << "読み取り失敗 : " << image_names[i] << endl;
				continue;
			}

			//反転処理
			vector<Point2f> points;
			if(f == 0){	//そのまま
				points = points_vecs[i];
			}
			else{
				//画像を反転
				flip(orl_image, orl_image, 1);
				//形状をを反転
				points = flied_points_vecs[i];
			}


			//ヒストグラムの正規化
			Mat normed_image;
			equalizeHist(orl_image, normed_image);
			//最大サイズのものを検出 (小さいものは検出しない等の設定)
			vector<Rect> face_rects;
			this->cascade.detectMultiScale(
					normed_image, face_rects,
					cascade_scale, cascade_min_neighbours,
					0 | CV_HAAR_FIND_BIGGEST_OBJECT |CV_HAAR_SCALE_IMAGE,
					cascade_min_size);
			
			//検出失敗時 continue
			if(face_rects.size() == 0) continue;

			//視覚化
// 			Mat canvas = orl_image.clone();
//     		for(int n = 0; n < points.size(); n++){
// 				circle(canvas, points[n], 1, Scalar(0,255,0), 2, CV_AA);
// 			}
// 			rectangle(canvas, face_rects[0], Scalar(255));
// 			imshow("Detector Training", canvas);
// 			waitKey(10); 


			//形状が検出枠に十分に入っている場合のみ使用
			if(this->isBoundingEnough(points, face_rects[i], bounding_per)){
				//形状の中心を取得
				Point2f center = this->calcMassCenter(points);
				float width = face_rects[0].width;

				//四角形の中心を計算
				Point2f rect_center = face_rects[0].tl() + face_rects[0].br();
				rect_center.x *= 0.5;
				rect_center.y *= 0.5;

				//差分を一時保存
				offset_x.push_back( (center.x - rect_center.x ) / width );
				offset_y.push_back( (center.y - rect_center.y ) / width );
				offset_scale.push_back(this->calcScaleForBase(points) / width);
			}
		}
	}
	//ソート
	Mat x, y, scale;
	cv::sort(Mat(offset_x), x, CV_SORT_EVERY_COLUMN|CV_SORT_ASCENDING);
	cv::sort(Mat(offset_y), y, CV_SORT_EVERY_COLUMN|CV_SORT_ASCENDING);
	cv::sort(Mat(offset_scale), scale, CV_SORT_EVERY_COLUMN|CV_SORT_ASCENDING);

	//中央値を取得、保存
	this->offsets = Vec3f(x.at<float>(x.rows/2),
						 y.at<float>(y.rows/2), scale.at<float>(scale.rows/2));
}
//検出
void Detector::detect(const Mat& image, vector<Point2f>& dst_points, const float cascade_scale, const int cascade_min_neighbours, const Size cascade_min_size){
	if(this->cascade.empty()){
		cerr << "cascadeが未初期化" << endl;
		return;
	}

	dst_points.clear();

	//グレーイメージを取得
	Mat gray_image;
	if(image.channels() == 1) gray_image = image;
	else cvtColor(image, gray_image, CV_RGB2GRAY);

	//ヒストグラムの正規化
	equalizeHist(gray_image, gray_image);
	//最大サイズのものを検出 (小さいものは検出しない等の設定)
	vector<Rect> face_rects;
	this->cascade.detectMultiScale(
			gray_image, face_rects,
			cascade_scale, cascade_min_neighbours,
			0 | CV_HAAR_FIND_BIGGEST_OBJECT |CV_HAAR_SCALE_IMAGE,
			cascade_min_size);

	//検出失敗時 return
	if(face_rects.size() == 0) return;


	int points_size = this->base_shape.size();
	//戻り値の領域確保
	dst_points.resize(points_size);


	//差分を計算
	Vec3f cur_offsets = this->offsets * face_rects[0].width;
	float shift_x = face_rects[0].x + 0.5*face_rects[0].width + cur_offsets[0];
	float shift_y = face_rects[0].y + 0.5*face_rects[0].height + cur_offsets[1];
	//各点に適応
	for(int i = 0; i < points_size; i++){
		dst_points[i].x = cur_offsets[2]*this->base_shape[i].x + shift_x;
		dst_points[i].y = cur_offsets[2]*this->base_shape[i].y + shift_y;
	}

	//再検出用の四角と画像を用意
	this->face_rect = face_rects[0];
	resize(image(face_rect), this->face_small, Size(), SMALL_IMAGE_SCALE, SMALL_IMAGE_SCALE, INTER_AREA);

	return;
}
//再検出
void Detector::redetect(const Mat& image, vector<Point2f>& points){
	//事前検出が行われていない場合
	if(this->face_small.empty() || this->face_rect.x == 0 || this->face_rect.y == 0){
		//初回検出実行
		this->detect(image, points);
		return;
	}

	//テンプレートマッチング
	Mat small_image;
	Mat response;
	resize(image, small_image, Size(), SMALL_IMAGE_SCALE, SMALL_IMAGE_SCALE, INTER_AREA);
	matchTemplate(small_image, this->face_small, response, CV_TM_CCOEFF_NORMED); 
	//合計を1にする
	normalize(response, response, 0, 1, NORM_MINMAX);
	response /= sum(response)[0];
	//responseの一番大きな値の場所を取得
	Point max_point;
	minMaxLoc(response, 0,0,0, &max_point);

	//画像外に領域が出たら終了
	if(image.cols <= max_point.x/SMALL_IMAGE_SCALE + this->face_rect.width ||
	   image.rows <= max_point.y/SMALL_IMAGE_SCALE + this->face_rect.height) return;

	float shift_x = max_point.x/SMALL_IMAGE_SCALE - this->face_rect.x;
	float shift_y = max_point.y/SMALL_IMAGE_SCALE - this->face_rect.y;
	//点をシフト
	for(int i = 0; i < points.size(); i++){
		points[i].x += shift_x;
		points[i].y += shift_y;
	}

	//再検出用の四角と画像を更新
	this->face_rect.x = max_point.x/SMALL_IMAGE_SCALE;
	this->face_rect.y = max_point.y/SMALL_IMAGE_SCALE;
	resize(image(face_rect), this->face_small, Size(), SMALL_IMAGE_SCALE, SMALL_IMAGE_SCALE, INTER_AREA);

// 	imshow("face_small", this->face_small);
}

//保存
void Detector::save(const string& filename){
	FileStorage cvfs(filename, CV_STORAGE_WRITE);
	
	//メンバ変数を保存
	write(cvfs, "Base_Shape", this->base_shape);
	write(cvfs, "Offsets", this->offsets);
}
//読み取り
void Detector::load(const string& filename){
	FileStorage cvfs(filename, CV_STORAGE_READ);
	FileNode node(cvfs.fs, NULL);

	//メンバ変数を読み取り
	read(node["Base_Shape"], this->base_shape);
	read(node["Offsets"], this->offsets, Vec3f());
}

//視覚化 (webcam使用)
void Detector::visualize(){
	VideoCapture cap(0);
	if(!cap.isOpened()) {
		cerr << "No Webcam." << endl;
		return;
	}

	Mat image;
	for(;;) {
		cap >> image;

		//検出
		vector<Point2f> points;
		this->detect(image, points);

		//描画
		for(int i = 0; i < points.size(); i++){
			circle(image, points[i], 1, Scalar(0,255,0), -1, 8, 0);
		}

		imshow("detect", image);
		if(waitKey(10) == 'q') break;
	}
}
//視覚化 (ファイル使用)
void Detector::visualize(const vector<string>& image_names){
	for(int i = 0; i < image_names.size(); i++){
		Mat image = imread(image_names[i], 0);

		//検出
		vector<Point2f> points;
		this->detect(image, points);
		if(points.size() == 0){
			cerr << "No faces" << endl;
			continue;
		}

		//描画
		for(int i = 0; i < points.size(); i++){
			circle(image, points[i], 1, Scalar(200), -1, 8, 0);
		}

		imshow("detect", image);
		if(waitKey(0) == 'q') break;
	}
}



//private
//点が四角形の中に十分に入っているか確認
bool Detector::isBoundingEnough(const vector<Point2f>& points, const Rect& rect, const float percent){

	int inside_points_num = 0;
	//各点について四角形の内部にあるか判定
	for(int i = 0; i < points.size(); i++){
		if(rect.contains(points[i])) inside_points_num++;
	}

	//指定割合以上の点が内部にあったらtrue
	if((float)inside_points_num / (float)points.size()) return true;
	else return false;
}
//点群の重心を計算
Point2f Detector::calcMassCenter(const vector<Point2f>& points){
	Point2f mean_point(0,0);
	for(int i = 0; i < points.size(); i++){
		mean_point += points[i];
	}
	mean_point.x /= points.size();
	mean_point.y /= points.size();

	return mean_point;
}
//基準形状に対する大きさを計算
float Detector::calcScaleForBase(const vector<Point2f>& points){
	//重心座標
	Point2f center = this->calcMassCenter(points);

	//重心を引きつつ、基準形状に対する大きさを計算
	float scale = 0, base = 0;
	for(int i = 0; i < points.size(); i++){
		scale += (points[i] - center).dot(this->base_shape[i]);
		base += this->base_shape[i].dot(this->base_shape[i]);
	}
	return scale / base;
}
