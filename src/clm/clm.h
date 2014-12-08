#ifndef CLM_H_141021
#define CLM_H_141021

#include "shape.h"
#include "patch.h"
#include "detector.h"

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

using namespace cv;
using namespace std;

//======Clm======
class Clm{
public:
	//コンストラクタ
	Clm();
	//コンストラクタ init付き
	Clm(const string& data_dir,
		const string& cascade_file);
	Clm(const string& shape_file,
		const string& patch_file,
		const string& detector_file,
		const string& cascade_file);
	//初期化
	void init(const string& data_dir,
			  const string& cascade_file);
	void init(const string& shape_file,
			  const string& patch_file,
			  const string& detector_file,
			  const string& cascade_file);
	//検出、追跡
	bool track(const Mat& image,
			   vector<Point2f>& dst_points,
			   const bool init_flag = false,
			   const bool use_redetect = true);


	//Shape,Patch,Detectorを学習して保存
	static void train(const vector<string>& image_names,
			   const vector<vector<Point2f> >& points_vecs,
			   const string& CASCADE_FILE,
			   const vector<int>& symmetry,
			   const vector<Vec2i>& connections,
			   const string& OUTPUT_DIR);

	//symmetryを利用して点群を左右反転
	static void getFlippedPointsVecs(const vector<vector<Point2f> >& src_vecs,
							  vector<vector<Point2f> >& dst_vecs,
							  const vector<string>& image_names,
							  const vector<int>& symmetry);

private:
	//CLMファイルのデータフォルダ内の名前
	static const string SHAPE_FILE_NAME;
	static const string PATCH_FILE_NAME;
	static const string DETECTOR_FILE_NAME;

	//CLM用クラス
	Shape shape;
	PatchContainer patch;
	Detector detector;

	//1フレーム前の座標群
	vector<Point2f> pre_points;
};

#endif
