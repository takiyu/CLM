#ifndef DETECTOR_H_141021
#define DETECTOR_H_141021

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

class Detector{
	
public:
	//コンストラクタ
	Detector();
	Detector(const string& cascade_file);
	//カスケードの初期化
	void initCascade(const string& cascade_file);
	//学習
	void train(const vector<Point2f>& base_shape,
			   const vector<string>& image_names,
			   const vector<vector<Point2f> >& points_vecs,
			   const vector<vector<Point2f> >& flied_points_vecs
										= vector<vector<Point2f> >(0),
			   const float cascade_scale = 1.1,
			   const int cascade_min_neighbours = 2,
			   const Size cascade_min_size = Size(30,30),
			   const float bounding_per = 0.8);
	//検出  失敗時は (dst_points.size() == 0)
	void detect(const Mat& image,
				vector<Point2f>& dst_points,
				const float cascade_scale = 1.1,
				const int cascade_min_neighbours = 2,
				const Size cascade_min_size = Size(30,30));
	void redetect(const Mat& image,
				  vector<Point2f>& points);

	//保存
	void save(const string& filename);
	//読み取り
	void load(const string& filename);

	//視覚化 (webcam使用)
	void visualize();
	//視覚化 (ファイル使用)
	void visualize(const vector<string>& image_names);

private:
	//カスケード
	CascadeClassifier cascade;
	//基準形状
	vector<Point2f> base_shape;
	//検出枠と基準形状のオフセット
	Vec3f offsets;
	//顔領域 (再検出に利用)
	static const float SMALL_IMAGE_SCALE;
	Rect face_rect;
	Mat face_small, pre_face_small;


	//点が四角形の中に十分に入っているか確認
	bool isBoundingEnough(const vector<Point2f>& points,
						  const Rect& rect,
						  const float percent);
	//点群の重心を計算
	Point2f calcMassCenter(const vector<Point2f>& points);
	//基準形状に対する大きさを計算
	float calcScaleForBase(const vector<Point2f>& points);
};

#endif
