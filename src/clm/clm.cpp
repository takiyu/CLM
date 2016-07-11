#include "clm.h"

#include "shape.h"
#include "patch.h"
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

//ファイル名
const string Clm::SHAPE_FILE_NAME = "shape.data";
const string Clm::PATCH_FILE_NAME = "patch.data";
const string Clm::DETECTOR_FILE_NAME = "detector.data";

//======Clm======
//コンストラクタ
Clm::Clm(){
}
//コンストラクタ init付き
Clm::Clm(const string& data_dir, const string& cascade_file){
	init(data_dir, cascade_file);
}
Clm::Clm(const string& shape_file, const string& patch_file, const string& detector_file, const string& cascade_file){
	init(shape_file, patch_file, detector_file, cascade_file);
}
//初期化
void Clm::init(const string& data_dir, const string& cascade_file){
	this->pre_points.clear();

	//ロード
	this->shape.load(data_dir + "/" + SHAPE_FILE_NAME);
	this->patch.load(data_dir + "/" + PATCH_FILE_NAME);
	this->detector.load(data_dir + "/" + DETECTOR_FILE_NAME);
	this->detector.initCascade(cascade_file);
}
void Clm::init(const string& shape_file, const string& patch_file, const string& detector_file, const string& cascade_file){
	this->pre_points.clear();

	//ロード
	this->shape.load(shape_file);
	this->patch.load(patch_file);
	this->detector.load(detector_file);
	this->detector.initCascade(cascade_file);
}
//検出、追跡
bool Clm::track(const Mat& image, vector<Point2f>& result_points, const bool init_flag, const bool use_redetect){
	
	//---点群を初期化---
	if(init_flag == true || pre_points.size() == 0){
		this->detector.detect(image, pre_points);
		//顔の検出に失敗したら終了
	}
	//---点群を再検出(平行移動)---
	else{
		if(use_redetect) this->detector.redetect(image, pre_points);
	}
	//顔の検出に失敗したら終了
	if(pre_points.size() == 0) return false;

	//---検出---
	Size sizes[] = { Size(21,21), Size(11,11), Size(5,5) };
	//サイズを変えて繰り返し
	for(int i = 0; i < 3; i++){
		//形状を表すパラメータ
		Mat shape_param;
		//パッチ探索
		this->patch.calcPeaks(image, pre_points, pre_points, sizes[i]);
		//形状モデルで表現
		shape_param = this->shape.getPraram(pre_points);
		this->shape.getShape(pre_points, shape_param);
	}

	//戻り値
	result_points = pre_points;
	return true;
}

//Shape,Patch,Detectorを学習して保存
void Clm::train(const vector<string>& image_names, const vector<vector<Point2f> >& points_vecs, const string& CASCADE_FILE, const vector<int>& symmetry, const vector<Vec2i>& connections, const string& OUTPUT_DIR){

	//======反転形状データの準備======
	vector<vector<Point2f> > flipped_points_vecs;
	getFlippedPointsVecs(points_vecs, flipped_points_vecs, image_names, symmetry);

	//======Shape======
	Shape shape;
	//オリジナルと反転形状を統合
	vector<vector<Point2f> > united_points_vecs = points_vecs;
	united_points_vecs.insert(united_points_vecs.end(),
			flipped_points_vecs.begin(), flipped_points_vecs.end());
	//形状行列を取得
	shape.train(united_points_vecs);
	//保存
	shape.save(OUTPUT_DIR + "/shape.data");
	//視覚化
	shape.visualize(connections);

	//平均形状を取得
	vector<Point2f> mean_shape;
	shape.getMeanShape(mean_shape, 100);

	//======Patch======
	//パッチを学習
	PatchContainer patch;
	patch.train(mean_shape, image_names, points_vecs, flipped_points_vecs);
	//保存
	patch.save(OUTPUT_DIR + "/patch.data");
	//視覚化
	patch.visualize();

	//======Detector======
	//検出器を学習
	Detector detector(CASCADE_FILE);
	detector.train(mean_shape, image_names, points_vecs, flipped_points_vecs);
	//保存
	detector.save(OUTPUT_DIR + "/detector.data");
	//視覚化
	detector.visualize();

	cout << "Training Finished" << endl;
}

//symmetryを利用して点群を左右反転
void Clm::getFlippedPointsVecs(const vector<vector<Point2f> >& src_vecs, vector<vector<Point2f> >& dst_vecs, const vector<string>& image_names, const vector<int>& symmetry){
	assert(src_vecs.size() == image_names.size());

	//戻り値の初期化
	dst_vecs.reserve(src_vecs.size());
	//反転して追加
	for(int i = 0; i < src_vecs.size(); i++){
		//対応する画像の幅を取得
		Mat tmp_mat = imread(image_names[i], 0);
		if(tmp_mat.empty()){
			cerr << "Read Error : " << image_names[i] << endl;
			continue;
		}
		int width = tmp_mat.cols;
		//反転して追加
		vector<Point2f> flipped_points;

		assert(symmetry.size() == src_vecs[i].size());
		//初期化
		flipped_points.resize(src_vecs[i].size());
		//反転して追加
		for(int j = 0; j < src_vecs[i].size(); j++){
			flipped_points[j].x = width - 1 - src_vecs[i][symmetry[j]].x;
			flipped_points[j].y = src_vecs[i][symmetry[j]].y;
		}
		dst_vecs.push_back(flipped_points);
	}
}
