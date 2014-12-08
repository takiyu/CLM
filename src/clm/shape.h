#ifndef SHAPE_H_141017
#define SHAPE_H_141017

#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

class Shape{

public:
	//コンストラクタ
	Shape();
	//形状を学習し、形状行列を取得
	void train(const vector< vector<Point2f> >& points_vecs,
			   const float EFF_EIGEN_PAR = 0.95,
			   const int MAX_EFF_IDX = 10);
	//平均形状を取得
	void getMeanShape(vector<Point2f>& points, const int width = 100);
	//パラメータを指定して形状を取得
	void getShape(vector<Point2f>& points, const Mat& param);
	//パラメータのひな形を取得
	Mat getPraram();
	//形状からパラメータを取得
	Mat getPraram(const vector<Point2f> points);
	//学習した形状の幅を取得
	float calcWidth();

	//保存
	void save(const string& filename);
	//読み取り
	void load(const string& filename);

	//視覚化して表示
	void visualize(const vector<Vec2i>& connections);

//static
	//全ての点をscale倍する
	static void resizePoints(vector<Point2f>& points, float scale);
	//全ての点をshiftする
	static void shiftPoints(vector<Point2f>& points, Point2f shift);

	/* 描画 */
	//connectionsを元に点群を描画
	static void drawPoints(Mat& image,
						   const vector<Point2f>& points,
						   const Scalar& color,
						   const int radius,
						   const vector<Vec2i>& connections);
	//点の集合を描画
	static void drawPoints(Mat& image,
						   const vector<Point2f>& points,
						   const Scalar& color,
						   const int radius);
	//点のインデックス付きで描画
	static void drawPointsWithIdx(Mat& image,
								  const vector<Point2f>& points,
								  const Scalar& color,
								  const int radius);
	//点のインデックス付きで、connectionsを元に描画
	static void drawPointsWithIdx(Mat& image,
								  const vector<Point2f>& points,
								  const Scalar& color,
								  const int radius,
								  const vector<Vec2i>& connections);

private:
	//点の個数(=combiner.row/2)
	int points_size;
	//形状行列
	Mat combiner;
	//パラメータの分散
	Mat param_varience;

	//型変換 vector<Point2f> to 1dim Mat(32FC1) (列が１データ)
	Mat cvtPointVecs2Mat32f(const vector<vector<Point2f> >& points_vecs);
	//プロクラステス解析
	Mat calcProcrustes(const Mat& points,
					   const int max_times = 100,
					   const float epsilon = 1e-6);
	//平均形状からの変換で各顔を表現するための基底を求める
	Mat calcRigidTransformationBasis(const Mat& src);
	//固有ベクトル群を計算
	//	EFF_EIGEN_PAR	: 有効な固有値の割合
	//	MAX_EFF_IDX		: 使用する固有ベクトルの最大次数
	Mat calcEigenVectors(const Mat& points_diff,
						 const float EFF_EIGEN_PAR,
						 const int MAX_EFF_IDX);

	//パラメータの各成分の分散を計算
	Mat calcParamVarience(const Mat& points_mat);
	//偏差でパラメータに制限を付ける (default:3 omega -> 2 omega)
	void crampParam(Mat& param, const float deviation_times = 3.0);

};

#endif
