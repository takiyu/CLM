#include "shape.h"

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

/*
 * <点の扱いについて>
 * Mat(32FC1)型の座標はshapeクラス内でのみ使用する
 * その他ではvector<Point2f>を使用し、shapeクラスからの取り出し時に変換作業が行われる
*/

//public
//コンストラクタ
Shape::Shape(){
}
//学習 形状行列を計算
void Shape::train(const vector< vector<Point2f> >& points_vecs, const float EFF_EIGEN_PAR, const int MAX_EFF_IDX){
			   
	//---点の個数の取得と確認---
	this->points_size = points_vecs[0].size();
	for(int i = 1; i < points_vecs.size(); i++){
		assert(points_vecs[i].size() == this->points_size); 
	}

	//---Matに変換---
	Mat orl_points_mat = cvtPointVecs2Mat32f(points_vecs);

	//---プロクラステス解析---
	Mat points_mat = calcProcrustes(orl_points_mat);

	//---平均形状からの変換で各顔を表現する基底を求める---
	//(basis * param = points)
	Mat basis = calcRigidTransformationBasis(points_mat);

	//パラメータを取得(平均形状から変換可能な分のみ)
	//(basis.t()*basis = Identity)
	Mat params = basis.t() * points_mat;
	//平均形状からの差分を求める(偏差: x - mean)
	Mat points_diff = points_mat - basis * params;

	//---固有ベクトル群を取得---
	Mat eigenvectors = calcEigenVectors(points_diff, EFF_EIGEN_PAR, MAX_EFF_IDX);

	//---basisとeigenvectorsを結合---  ( eff_idx == eigenvectors.cols )
	this->combiner.create(points_mat.rows, eigenvectors.cols+4, CV_32F);
	basis.copyTo(this->combiner(Rect(0, 0, 4, basis.rows)));
	eigenvectors.copyTo(this->combiner(Rect(4, 0, eigenvectors.cols, eigenvectors.rows)));

	//---分散を計算---
	this->param_varience = this->calcParamVarience(orl_points_mat);

	return;
}
//平均形状を取得
void Shape::getMeanShape(vector<Point2f>& points, const int width){
	Mat mean_param = this->getPraram();
	mean_param.at<float>(0) = width / this->calcWidth();
	this->getShape(points, mean_param);
}
//パラメータを指定して形状を取得
void Shape::getShape(vector<Point2f>& points, const Mat& param){
	points.clear();
	points.reserve(this->points_size);
	//パラメータから形状を計算
	Mat paramed_shape = this->combiner * param;
	//Mat -> vector<Point2f>
	for(int i = 0; i < paramed_shape.rows/2; i++){
		points.push_back( Point2f(paramed_shape.at<float>(2*i), paramed_shape.at<float>(2*i+1))	);
	}
}
//パラメータのひな形を取得
Mat Shape::getPraram(){
	return Mat::zeros(this->combiner.cols, 1, CV_32F);
}
//形状からパラメータを取得
Mat Shape::getPraram(const vector<Point2f> points){
	assert(this->points_size == points.size());

	//vector<point2f> -> Mat
	Mat points_mat(2*this->points_size, 1, CV_32F);
	//列に１データを保存
	Mat y = Mat(points).reshape(1, 2*this->points_size);
	y.copyTo(points_mat);
	
	Mat param = this->combiner.t() * points_mat;

	//分散による上限 3omega
	this->crampParam(param);

	return param;
}
//学習した形状の幅を取得
float Shape::calcWidth(){
	//x座標の幅を計算
	float x_max = combiner.at<float>(0,0);
	float x_min = combiner.at<float>(0,0);
	for(int i = 1; i < combiner.rows/2; i++){
		x_max = max(x_max, combiner.at<float>(2*i,0));
		x_min = min(x_min, combiner.at<float>(2*i,0));
	}
	return x_max - x_min;
}

//保存
void Shape::save(const string& filename){
	FileStorage cvfs(filename, CV_STORAGE_WRITE);
	write(cvfs, "Combiner", this->combiner);
	write(cvfs, "ParamVarience", this->param_varience);
}
//読み取り
void Shape::load(const string& filename){
	FileStorage cvfs(filename, CV_STORAGE_READ);
	FileNode node(cvfs.fs, NULL);
	read(node["Combiner"], this->combiner);
	this->points_size = this->combiner.rows / 2;
	read(node["ParamVarience"], this->param_varience);
}

//視覚化して表示
void Shape::visualize(const vector<Vec2i>& connections){
	const Scalar COLOR(255);
	const string WINDOW_NAME = "visualized_shape";

	//パラメータの準備
	Mat param = this->getPraram();
	//スケールと座標を設定
	param.at<float>(0) = 200.0f / this->calcWidth();//scale
	param.at<float>(2) = 1300;//dx
	param.at<float>(3) = 1300;//dy

	//パラメータ用の連続変化する変数を用意
	vector<float> val;
	for(int i = 0; i < 50; i++) val.push_back(float(i)/50);
	for(int i = 0; i < 50; i++) val.push_back(float(50-i)/50);
	for(int i = 0; i < 50; i++) val.push_back(-float(i)/50);
	for(int i = 0; i < 50; i++) val.push_back(-float(50-i)/50);

	//パラメータの値を変更しながら描画開始
	while(true){
		//スケールと座標は変化させない
		for(int n = 4; n < param.rows; n++){
			for(int m = 0; m < val.size(); m++){
				param.at<float>(n) = val[m] * 100;
				//パラメータに制限を与える
				this->crampParam(param);
				//パラメータから形状を計算
				vector<Point2f> paramed_points;
				this->getShape(paramed_points, param);

				//表示する画像
				Mat canvas = Mat::zeros(300,300,CV_32F);

				//汎用関数で描画
				Shape::drawPoints(canvas, paramed_points, COLOR, 1, connections);

				//表示
				imshow(WINDOW_NAME, canvas);

				//qキーで終了
				char key = waitKey(10);
				if(key == 'q'){
					destroyWindow(WINDOW_NAME);
					return;
				}
			}
		}
	}
}

//static
//全ての点をscale倍する
void Shape::resizePoints(vector<Point2f>& points, float scale){
	for(int i = 0; i < points.size(); i++){
		points[i] *= scale;
	}
}
//全ての点をshiftする
void Shape::shiftPoints(vector<Point2f>& points, Point2f shift){
	for(int i = 0; i < points.size(); i++){
		points[i] += shift;
	}
}

/* 描画 */
//connectionsを元に点群を描画
void Shape::drawPoints(Mat& image, const vector<Point2f>& points, const Scalar& color, const int radius, const vector<Vec2i>& connections){
	//点を描画
	for(int i = 0; i < points.size(); i++) {
		circle(image, points[i], radius, color, -1, 8, 0);
	}
	//線を描画
	for(int j = 0; j < connections.size(); j++){
		line(image, points[connections[j][0]], points[connections[j][1]], color, radius);
	}
}
//点の集合を描画
void Shape::drawPoints(Mat& image, const vector<Point2f>& points, const Scalar& color, const int radius){
	//描画
	for(int i = 0; i < points.size(); i++) {
		circle(image, points[i], radius, color, -1, 8, 0);
		if(i != 0) line(image, points[i-1], points[i], color, radius);
	}
	line(image, points[0], points[points.size()-1], color, radius);
}
//点のインデックス付きで描画
void Shape::drawPointsWithIdx(Mat& image, const vector<Point2f>& points, const Scalar& color, const int radius){
	drawPoints(image, points, color, radius);
	//描画
	for(int i = 0; i < points.size(); i++) {
		stringstream ss;
		ss << i;
		//青色
		putText(image, ss.str(), points[i], FONT_HERSHEY_COMPLEX_SMALL, 0.5, color);
	}
}
//点のインデックス付きで、connectionsを元に描画
void Shape::drawPointsWithIdx(Mat& image, const vector<Point2f>& points, const Scalar& color, const int radius, const vector<Vec2i>& connections){
	drawPoints(image, points, color, radius, connections);
	//描画
	for(int i = 0; i < points.size(); i++) {
		stringstream ss;
		ss << i;
		//青色
		putText(image, ss.str(), points[i], FONT_HERSHEY_COMPLEX_SMALL, 0.5, color);
	}
}



//private
//型変換 vector<Point2f> to 1dim Mat(32FC1) (列が１データ)
Mat Shape::cvtPointVecs2Mat32f(const vector<vector<Point2f> >& points_vecs){
	int vec_size = points_vecs.size();
	assert(vec_size > 0);

	//コピー
	Mat dst(2*this->points_size, vec_size, CV_32F);
	for(int i = 0; i < vec_size; i++){
		//列に１データを保存
		Mat y = Mat(points_vecs[i]).reshape(1,2*this->points_size);
		y.copyTo(dst.col(i));
	}

	return dst;
}
//プロクラステス解析
Mat Shape::calcProcrustes(const Mat& points, const int max_times, const float epsilon){
	int vec_size = points.cols;

	//重心を取り除く
	Mat dst_points = points.clone();
	for(int i = 0; i < vec_size; i++){
		//重心を求める
		float mean_x = 0, mean_y = 0;
		for(int j = 0; j < this->points_size; j++){
			mean_x += dst_points.at<float>(2*j  ,i);
			mean_y += dst_points.at<float>(2*j+1,i);
		}
		mean_x /= this->points_size;
		mean_y /= this->points_size;
		//重心を引く
		for(int j = 0; j < this->points_size; j++){
			dst_points.at<float>(2*j  ,i) -= mean_x;
			dst_points.at<float>(2*j+1,i) -= mean_y;
		}
	}

	//optimise scale and rotation
	Mat pre_mean_shape;
	for(int n = 0; n < max_times; n++){
		//平均形状を求める mean_shape
		Mat mean_shape = dst_points * Mat::ones(vec_size,1,CV_32F) / vec_size;
		normalize(mean_shape, mean_shape);
		//誤差が一定以下になったら終了
		if(n != 0){
			if(norm(mean_shape, pre_mean_shape) < epsilon) break;
		}

		//一つ前の平均形状を更新
		pre_mean_shape = mean_shape.clone();

		for(int i = 0; i < vec_size; i++){
			//各形状から平均形状への変換行列を求める
			float a = 0, b = 0, d = 0;
			for(int j = 0; j < this->points_size; j++){
				float src_0 = dst_points.at<float>(2*j  ,i);
				float src_1 = dst_points.at<float>(2*j+1,i);
				float dst_0 = mean_shape.at<float>(2*j  ,0);
				float dst_1 = mean_shape.at<float>(2*j+1,0);
				d += src_0 * src_0 + src_1 * src_1;
				a += src_0 * dst_0 + src_1 * dst_1;
				b += src_0 * dst_1 - src_1 * dst_0;
			}
			a /= d;
			b /= d;
			//行列(a,-b, b,a)
			for(int j = 0; j < this->points_size; j++){
				float x = dst_points.at<float>(2*j  , i);
				float y = dst_points.at<float>(2*j+1, i);
				dst_points.at<float>(2*j  , i) = a*x - b*y;
				dst_points.at<float>(2*j+1, i) = b*x + a*y;
			}
		}
	}
	return dst_points;
}
//平均形状からの変換で各顔を表現するための基底を求める
Mat Shape::calcRigidTransformationBasis(const Mat& src){
	int vec_size = src.cols;
	//平均形状を求める
	Mat mean_shape = src * Mat::ones(vec_size,1,CV_32F) / vec_size;
	
	//基底
	Mat basis(2*this->points_size, 4, CV_32F);
	for(int i = 0; i < this->points_size; i++){
		basis.at<float>(2*i, 0) = mean_shape.at<float>(2*i  );
		basis.at<float>(2*i, 1) =-mean_shape.at<float>(2*i+1);
		basis.at<float>(2*i, 2) = 1.0f;
		basis.at<float>(2*i, 3) = 0.0f;

		basis.at<float>(2*i+1, 0) = mean_shape.at<float>(2*i+1);
		basis.at<float>(2*i+1, 1) = mean_shape.at<float>(2*i  );
		basis.at<float>(2*i+1, 2) = 0.0f;
		basis.at<float>(2*i+1, 3) = 1.0f;
	}

	//basisにシュミットの直交化を適用
	for(int i = 0; i < 4; i++){
		Mat v = basis.col(i);
		for(int j = 0; j < i; j++){
			Mat w = basis.col(j);
			v -= w * (w.t() * v);//v-=w * (内積)
		}
		normalize(v, v);
	}

	return basis;
}
//固有ベクトル群を計算
Mat Shape::calcEigenVectors(const Mat& points_diff, const float EFF_EIGEN_PAR, const int MAX_EFF_IDX){
	//特異値分解 (共分散行列 : diff*diff.t())
	SVD svd(points_diff * points_diff.t());

	//固有値の合計を計算
	float eigenvalue_sum = 0;
	for(int i = 0; i < svd.w.rows; i++){
		eigenvalue_sum += svd.w.at<float>(i,0);
	}

	//有効な次数の上限 (指定、データ数-1、点の数-1)の最小値
	int max_idx = min(MAX_EFF_IDX, min(points_diff.cols-1, this->points_size-1));

	//有効な次数(eff_idx)を計算
	int eff_idx = 0;
	float tmp_sum = 0;
	for(; eff_idx < max_idx; eff_idx++){
		tmp_sum += svd.w.at<float>(eff_idx,0);
		//一定割合を超えたら終了
		if(tmp_sum / eigenvalue_sum >= EFF_EIGEN_PAR){
			//idxを一つ進める (max_idxが最大)
			if(eff_idx < max_idx-1) eff_idx++;
			break;
		}
	}
	//使用する固有ベクトル群を取得 (svd.u.rows == points_mat.rows)
	return svd.u(Rect(0,0,eff_idx,svd.u.rows));
}
//パラメータの各成分の分散を計算
Mat Shape::calcParamVarience(const Mat& points_mat){
	//パラメータ群を計算
	Mat params = this->combiner.t() * points_mat;
	//パラメータのスケールで割る
	for(int i = 0; i < params.cols; i++){
		params.col(i) /= params.at<float>(0,i);
	}

	//分散を保存する行列
	Mat variance(params.rows, 1, CV_32F);

	//2乗 (全て正になる)
	pow(params, 2, params);
	//分散を計算するための行列
	Mat ones_to_var = Mat::ones(1, params.cols, CV_32F) / (float)(params.cols-1);
	//各成分について分散を計算
	for(int i = 0; i < variance.rows; i++){
		if(i < 4) variance.at<float>(i) = -1;
		else variance.at<float>(i) = params.row(i).dot(ones_to_var);
	}

	return variance;
}
//偏差でパラメータに制限を付ける
void Shape::crampParam(Mat& param, const float deviation_times){
	float scale = param.at<float>(0);

	//正規化
	Mat normed_param = abs(param);
	//パラメータ部分のみ計算
	for(int i = 4; i < this->param_varience.rows; i++){
		//偏差*係数
		float dev_x = scale * deviation_times * sqrtf(this->param_varience.at<float>(i)); 
		//dev_xを上回る場合は更新
		if(normed_param.at<float>(i) > dev_x){
			if(param.at<float>(i) > 0) param.at<float>(i) = dev_x;
			else 					   param.at<float>(i) = -1 * dev_x;
		}
	}
}
