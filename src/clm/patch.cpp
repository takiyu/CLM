#include "patch.h"

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


//===PatchCell===
//public
//コンストラクタ
PatchCell::PatchCell(){
}
//学習
void PatchCell::train(const vector<Mat>& training_images, const Size patch_size, const int training_count, const float ideal_map_variance, const float weight_init, const float train_fetter){
	int images_num = training_images.size();
	int patch_pixel_num = patch_size.width * patch_size.height;

	//画像とパッチのサイズ確認
	Size training_size = training_images[0].size();
	assert(training_size.width >= patch_size.width);
	assert(training_size.height >= patch_size.height);

	//理想的なresponse_mapを作成
	int map_width = training_size.width - patch_size.width;
	int map_height = training_size.height - patch_size.height;
	Mat ideal_map(map_width, map_height, CV_32F);
	//ideal_mapの値を計算 (中央が山の正規分布)
	for(int y = 0; y < map_height; y++){
		float dist_y = (map_height-1)/2 - y;//中心までの距離
		for(int x = 0; x < map_width; x++){
			float dist_x = (map_width-1)/2 - x;//中心までの距離
			//正規分布を計算
			ideal_map.at<float>(y,x) =
				exp( -1*(dist_x*dist_x + dist_y*dist_y) / (ideal_map_variance*2) );
		}
	}
	//0~1の間に正規化
	normalize(ideal_map, ideal_map, 0, 1, NORM_MINMAX);


	//パッチの初期化
	this->patch = Mat::zeros(patch_size, CV_32F);
	//パッチ更新用行列
	Mat patch_diff(patch_size, CV_32F);
	//学習パッチサイズ画像の平均を求める行列
	Mat ones_to_avg = Mat::ones(patch_size, CV_32F) / patch_pixel_num;

	//学習画像の重み付け係数 (TODO doubleにするべきか)
	float weight = weight_init;
	//重み付け係数の更新係数 (TODO doubleにするべきか)
	float weight_step = pow(1e-8/weight, 1.0/training_count);

	//乱数生成器の初期化 (画像idxの取得に使用)
	RNG random(getTickCount());


	//指定個の画像について学習開始
	for(int n = 0; n < training_count; n++){
		//ランダムに学習画像を取得し、logを取る (入力の順番による偏りを排除)
		int i = random.uniform(0, images_num);
		Mat log_image = this->cvtLogImage(training_images[i]);

		//パッチ更新分を計算
		patch_diff = 0.0;
		for(int y = 0; y < map_height; y++){
			for(int x = 0; x < map_width; x++){
				//画像からパッチ学習領域を切り出し
				Mat training_patch = log_image( Rect(x, y, patch_size.width, patch_size.height) ).clone();
				training_patch -= training_patch.dot(ones_to_avg);//平均を引く
				normalize(training_patch, training_patch);
				
				//理想と現在のresponseを計算
				float real_response_dot = this->patch.dot(training_patch);
				float ideal_respoinse_dot = ideal_map.at<float>(y, x);
				//学習データの重みに差分を利用
				patch_diff += (ideal_respoinse_dot - real_response_dot) * training_patch; 
			}
		}

		//パッチの更新 (patch_diffが大きくなり過ぎないように足枷を付ける)
		this->patch += weight * (patch_diff - train_fetter*this->patch);
		//重みの更新 (小さくして画像ごとの重みの差を小さくする)
		weight *= weight_step;

		//視覚化 (calcResponseと同じ計算)
// 		Mat response;
// 		matchTemplate(log_image, this->patch, response, CV_TM_CCOEFF_NORMED);
// 		Mat normed_patch;
// 		normalize(this->patch, normed_patch, 0, 1, NORM_MINMAX);
// 		normalize(patch_diff, patch_diff, 0, 1, NORM_MINMAX);
// 		normalize(response, response, 0, 1, NORM_MINMAX);
// 		imshow("patch", normed_patch);
// 		imshow("patch_diff", patch_diff);
// 		imshow("response", response); 

		if(waitKey(10) == 'q') break;
	}
	return;
}
//パッチに対するresponseを取得
Mat PatchCell::calcResponse(const Mat& image){
	Mat log_image = this->cvtLogImage(image);

	Mat response;
	matchTemplate(log_image, this->patch, response, CV_TM_CCOEFF_NORMED); 

	//合計を1にする
	normalize(response, response, 0, 1, NORM_MINMAX);
	response /= sum(response)[0];

	return response;
}

//private
//画像を差異が少ないように変換 (Logをとる)
Mat PatchCell::cvtLogImage(const Mat &image){
	Mat dst; 

	//1chに変換
	if(image.channels() == 3) cvtColor(image, dst, CV_RGB2GRAY);
	else dst = image;
	assert(dst.channels() == 1);

	//CV_32Fに変換
	if(dst.type() != CV_32F) dst.convertTo(dst, CV_32F); 

	//logを取る
	dst += 1.0;
	log(dst, dst);

	return dst;
}




//===PatchContainer===
//public
//コンストラクタ
PatchContainer::PatchContainer(){
}
//学習
void PatchContainer::train(const vector<Point2f>& base_shape, const vector<string>& image_names, const vector<vector<Point2f> >& points_vecs, const vector<vector<Point2f> >& flied_points_vecs, const Size patch_size, const Size search_size){
	assert(points_vecs.size() == image_names.size());

	//反転を計算するかのフラグ
	bool flip_flag = (flied_points_vecs.size() == points_vecs.size());
	if(flip_flag) cout << "PatchContainer : flip_flag is true." << endl;
	else cout << "PatchContainer : flip_flag is false." << endl;

	//基準形状を保存
	this->base_shape = base_shape;
	//形状を構成する点の個数を保存
	this->points_size = base_shape.size();
	//学習イメージデータの大きさを設定
	Size window_size = patch_size + search_size;

	//パッチの初期化と領域確保
	this->patches.clear();
	this->patches.resize(this->points_size);


	//パッチ用データを作成し、学習開始
	for(int i = 0; i < points_size; i++){
		cout << "Training " << i+1 << "th patch" << endl;

		//変換した学習データ群を保存するvector<Mat>
		vector<Mat> training_images;
		training_images.reserve(image_names.size());//反転分を含めて2倍

		//オリジナルと反転分をそれぞれ計算
		for(int f = 0; f < (flip_flag ? 2 : 1); f++){

			//各データのパッチ領域を切り出し
			for(int j = 0; j < image_names.size(); j++){

				//画像読み取り
				Mat orl_image = imread(image_names[j], 0);
				if(orl_image.empty()){
					cout << "読み取り失敗 : " << image_names[j] << endl;
					continue;
				}

				//反転処理
				vector<Point2f> points;
				if(f == 0){	//そのまま
					points = points_vecs[j];
				}
				else{
					//画像を反転
					flip(orl_image, orl_image, 1);
					//形状をを反転
					points = flied_points_vecs[j];
				}

				//アフィン変換
				//基準形状からの変換行列を取得 (平行移動成分は使用しない)
				Mat aff_from_base = this->calcAffineFromBase(points);
				//パッチ領域からの平行移動成分を計算
				this->setAffineRotatedTranslation(aff_from_base, points[i], window_size);

				//基準形状へ変換し、パッチ領域を取得 (aff_from_baseなのでinverse)
				Mat warped_image;
				warpAffine(orl_image, warped_image, aff_from_base, window_size, INTER_LINEAR+WARP_INVERSE_MAP);

				//追加
				training_images.push_back(warped_image);
			}
		}
		//パッチ学習
		this->patches[i].train(training_images, patch_size);
	}
	return;
}
//最もパッチの反応が大きい点を計算
void PatchContainer::calcPeaks(const Mat& src_image, const vector<Point2f>& src_points, vector<Point2f>& dst_points, const Size search_size){
	assert(this->points_size == src_points.size());

	//基準形状からの変化行列
	Mat aff_from_base = this->calcAffineFromBase(src_points);
	//基準形状への変化行列
	Mat aff_to_base = this->calcInverseAffine(aff_from_base);


	//基準形状へ入力形状を変換
	vector<Point2f> based_points;
	this->applyAffineToPoints(src_points, aff_to_base, based_points);

	//各々の点に適応する変換行列
	Mat each_point_aff = aff_from_base.clone();
	//各点と領域について
	for(int i = 0; i < this->points_size; i++){
		//使用する領域サイズ
		Size window_size = this->patches[i].getPatchSize() + search_size;

		//パッチ領域からの平行移動成分を計算
		this->setAffineRotatedTranslation(each_point_aff, src_points[i], window_size);

		//基準形状へ領域を変換 (aff_from_baseを利用しているのでinverse)
		Mat warped_image;
		warpAffine(src_image, warped_image, each_point_aff, window_size, INTER_LINEAR+WARP_INVERSE_MAP);

		//パッチに対するresponseを取得
		Mat response = this->patches[i].calcResponse(warped_image);

		//responseの一番大きな値の場所を取得
		Point max_point;
		minMaxLoc(response, 0,0,0, &max_point);

		//responseを適応
		based_points[i].x += max_point.x - 0.5*search_size.width; 
		based_points[i].y += max_point.y - 0.5*search_size.height; 
	}

	//基準形状に変換した点群を逆変換
	this->applyAffineToPoints(based_points, aff_from_base, dst_points);

	return;
}

//保存
void PatchContainer::save(const string& filename){
	FileStorage cvfs(filename, CV_STORAGE_WRITE);
	
	//メンバ変数を保存
	write(cvfs, "Base_Shape", this->base_shape);
	write(cvfs, "Points_Size", this->points_size);

	//パッチの個数を保存
	write(cvfs, "Patches_Num", (int)this->patches.size());
	//パッチ本体の内容を保存
	for(int i = 0; i < this->patches.size(); i++){
		stringstream ss;
		ss << "Patch" << i;
		write(cvfs, ss.str(), this->patches[i].getPatch());
	}
	return;
}
//読み取り
void PatchContainer::load(const string& filename){
	FileStorage cvfs(filename, CV_STORAGE_READ);
	FileNode node(cvfs.fs, NULL);

	//メンバ変数を読み取り
	read(node["Base_Shape"], this->base_shape);
	read(node["Points_Size"], this->points_size, 0);

	//パッチの個数を取得
	int patches_size;
	read(node["Patches_Num"], patches_size, 0);
	//パッチ本体の個数をセット
	this->patches.resize(patches_size);

	//それぞれのパッチを読み取り
	for(int i = 0; i < this->patches.size(); i++){
		stringstream ss;
		ss << "Patch" << i;
		Mat patch_mat;
		//読み取り
		read(node[ss.str()], patch_mat);
		//設定
		this->patches[i].setPatch(patch_mat);
	}

	//エラー出力
	if(this->points_size == 0){	cerr << "Points_Sizeの読み取り失敗" << endl; }
	if(this->patches.size() == 0){ cerr << "Patchの読み取り失敗" << endl; }

	return;
}

//視覚化
void PatchContainer::visualize(){

	//パッチを取得し、表示用に変換
	vector<Mat> normed_patch(this->patches.size());
	for(int i = 0; i < this->points_size; i++){
		normed_patch[i] = this->patches[i].getPatch();
		normalize(normed_patch[i], normed_patch[i], 0, 255, CV_MINMAX);
	}

	//点の平行移動距離を計算
	Rect bounding_rect = boundingRect(this->base_shape);
	Point shift_amount = -1 * bounding_rect.tl() * 1.4;

	Mat canvas = Mat::zeros(400, 400, CV_8UC1);


	//表示開始
	while(true){
		for(int i = 0; i < this->points_size; i++){

			normed_patch[i].copyTo(canvas(
						Rect(this->base_shape[i].x + shift_amount.x,
							 this->base_shape[i].y + shift_amount.y,
							 normed_patch[i].cols, normed_patch[i].rows)));

			//表示
			imshow("Visualized_Patches", canvas);
			if(waitKey(10) == 'q') return;
		}
	}
}


//private
//base_shapeからの変形を表すアフィン行列を計算 (形状の差異は吸収)
Mat PatchContainer::calcAffineFromBase(const vector<Point2f>& points){
	//base_shapeは既に原点にあると仮定
	assert(points.size() == this->points_size);

	//dstの重心を求める
	float dst_x_mean = 0, dst_y_mean = 0;
	for(int i = 0; i < this->points_size; i++){
		dst_x_mean += points[i].x;
		dst_y_mean += points[i].y;
	}
	dst_x_mean /= this->points_size;
	dst_y_mean /= this->points_size;

	//重心が原点に移動した変換先形状
	vector<Point2f> dst_points(this->points_size);
	//重心を引き、原点を中央にする
	for(int i = 0; i < this->points_size; i++){
		dst_points[i].x = points[i].x - dst_x_mean;
		dst_points[i].y = points[i].y - dst_y_mean;
	}

	//scaleとrotationを計算
	float a = 0, b = 0, d = 0;
	for(int i = 0; i < this->points_size; i++){
		float src_0 = this->base_shape[i].x;
		float src_1 = this->base_shape[i].y;
		float dst_0 = dst_points[i].x;
		float dst_1 = dst_points[i].y;
		d += src_0 * src_0 + src_1 * src_1;
		a += src_0 * dst_0 + src_1 * dst_1;
		b += src_0 * dst_1 - src_1 * dst_0;
	}
	a /= d;
	b /= d;

	return (Mat_<float>(2,3) << a, -b, dst_x_mean,
		   						b,  a, dst_y_mean);
}
//領域が回転した場合への平行移動成分を計算
void PatchContainer::setAffineRotatedTranslation(Mat& affine_mat, const Point2f& base_point, const Size& window_size){
	//点座標 - 回転でずれた分
	affine_mat.at<float>(0,2) = base_point.x -
		(affine_mat.at<float>(0,0) * (window_size.width-1)/2 +
		 affine_mat.at<float>(0,1) * (window_size.height-1)/2);
	affine_mat.at<float>(1,2) = base_point.y -
		(affine_mat.at<float>(1,0) * (window_size.width-1)/2 +
		 affine_mat.at<float>(1,1) * (window_size.height-1)/2);
}
//アフィン変換行列の逆変換行列を計算
Mat PatchContainer::calcInverseAffine(const Mat& affine_mat){
	Mat dst_affine(2, 3, CV_32F);
	//行列式
	float det = affine_mat.at<float>(0,0)*affine_mat.at<float>(1,1)
			  - affine_mat.at<float>(1,0)*affine_mat.at<float>(0,1);
	//回転成分を計算
	dst_affine.at<float>(0,0) = affine_mat.at<float>(1,1) / det;
	dst_affine.at<float>(1,1) = affine_mat.at<float>(0,0) / det;
	dst_affine.at<float>(0,1) = -affine_mat.at<float>(0,1) / det;
	dst_affine.at<float>(1,0) = -affine_mat.at<float>(1,0) / det;

	//平行移動成分を回転して符号を逆にする
	Mat rot = dst_affine(Rect(0, 0, 2, 2));
	Mat trans = -1 * rot * affine_mat.col(2);
	//貼り付け
	trans.copyTo(dst_affine.col(2));

	return dst_affine;
}
//アフィン変換を点群に適用
void PatchContainer::applyAffineToPoints(const vector<Point2f>& src_points, const Mat& aff_mat, vector<Point2f>& dst_points){
	dst_points.resize(src_points.size());

	for(int i = 0; i < dst_points.size(); i++){
		dst_points[i].x = aff_mat.at<float>(0,0)*src_points[i].x
						+ aff_mat.at<float>(0,1)*src_points[i].y
						+ aff_mat.at<float>(0,2);
		dst_points[i].y = aff_mat.at<float>(1,0)*src_points[i].x
						+ aff_mat.at<float>(1,1)*src_points[i].y
						+ aff_mat.at<float>(1,2);
	}
	return;
}

