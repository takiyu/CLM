#ifndef PATCH_H_141021
#define PATCH_H_141021

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


//===PatchCell===
//一つのパッチを表すクラス
class PatchCell{
public:
	//コンストラクタ
	PatchCell();

	//学習
	void train(const vector<Mat>& training_images,
			   const Size patch_size,
			   const int Training_count = 1000,
			   const float ideal_map_variance = 1.0,
			   const float weight_init = 1e-3,
			   const float train_fetter = 1e-6);
	//パッチに対するresponseを取得
	Mat calcResponse(const Mat& image);

	//パッチを取得
	Mat getPatch(){	return this->patch.clone();	}
	//パッチを設定
	void setPatch(const Mat& src){ this->patch = src.clone(); return; }
	//パッチサイズを取得
	Size getPatchSize(){ return this->patch.size(); }

private:
	//パッチ本体
	Mat patch;

	//画像を差異が少ないように変換 (Logをとる)
	Mat cvtLogImage(const Mat& image);

};



//===PatchContainer===
//PatchCellを統括するクラス
class PatchContainer{
public:
	//コンストラクタ
	PatchContainer();

	//学習
	// base_shape : 学習の基準となる形状　平均形状を想定
	// flied_points_vecs : 反転した形状データ　反転分を計算しない場合は無視する
	void train(const vector<Point2f>& base_shape,
			   const vector<string>& image_names,
			   const vector<vector<Point2f> >& points_vecs,
			   const vector<vector<Point2f> >& flied_points_vecs
												= vector<vector<Point2f> >(0),
			   const Size patch_size = Size(11,11),
			   const Size search_size = Size(11,11));
	//最もパッチの反応が大きい点を計算
	void calcPeaks(const Mat& src_image,
				   const vector<Point2f>& src_points,
				   vector<Point2f>& dst_points,
				   const Size search_size = Size(21,21));

	//保存
	void save(const string& filename);
	//読み取り
	void load(const string& filename);

	//視覚化
	void visualize();

private:
	//基準となる形状データ
	vector<Point2f> base_shape;
	//１形状を構成する点の数 (==base_shape.size())
	int points_size;
	//パッチ本体
	vector<PatchCell> patches;

	//base_shapeからの変形を表すアフィン行列を計算 (形状の差異は吸収)
	Mat calcAffineFromBase(const vector<Point2f>& points);
	//領域が回転した場合への平行移動成分を計算
	void setAffineRotatedTranslation(Mat& affine_mat,
									 const Point2f& base_point,
									 const Size& window_size);
	//アフィン変換行列の逆変換行列を計算
	Mat calcInverseAffine(const Mat& affine_mat);

	void applyAffineToPoints(const vector<Point2f>& src_points,
							 const Mat& aff_mat,
							 vector<Point2f>& dst_points);

};

#endif
