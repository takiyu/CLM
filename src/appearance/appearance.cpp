#include "appearance.h"
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;


//コンストラクタ
App::App(){
}
//初期化付きコンストラクタ
App::App(const vector<Point2f> init_shape){
	init(init_shape);
}
//平面分割の初期化
void App::init(const vector<Point2f> init_shape){
	makeTriangleIdxMap(init_shape, this->triangle_map);
}

//三角形でwarp
void App::warpTriangle(const Mat& src_image, Mat& dst_image, const Point2f src_tri[], const Point2f dst_tri[]){
// 	if(dst_image.empty()) dst_image = src_image.clone();

	//アフィン変換行列を取得
	Mat aff_mat = getAffineTransform(src_tri, dst_tri);

	//アフィン変換
	Mat affed_image;
	warpAffine(src_image, affed_image, aff_mat, Size(dst_image.cols, dst_image.rows));


	vector<Point> mask_points(3);
	for(int i = 0; i < 3; i++){
		mask_points[i] = dst_tri[i];
	}

	//くり抜き用マスク
	Mat mask;
	if(dst_image.channels() == 3){
		mask = Mat::zeros(dst_image.rows, dst_image.cols, CV_8UC3);
		fillConvexPoly(mask, mask_points, Scalar(255,255,255));
	}else if(dst_image.channels() == 1){
		mask = Mat::zeros(dst_image.rows, dst_image.cols, CV_8UC1);
		fillConvexPoly(mask, mask_points, Scalar(255));
	}
	
	//くり抜き
	affed_image.copyTo(dst_image, mask);

// 	imshow("dst", dst_image);
// 	waitKey();
}
//三角形を取得してwarp
void App::warp(const Mat& src_image, Mat& dst_image, const vector<Point2f>& src_points, const vector<Point2f>& dst_points){
	for(int i = 0; i < this->triangle_map.size(); i++){
		Point2f src_tri[3];
		Point2f dst_tri[3];
		for(int j = 0; j < 3; j++){
			src_tri[j] = src_points[ this->triangle_map[i][j] ];
			dst_tri[j] = dst_points[ this->triangle_map[i][j] ];
		}
		this->warpTriangle(src_image, dst_image, src_tri, dst_tri);
	}
}

//三角形でwarp(gpu)
void App::warpTriangleGpu(const gpu::GpuMat& src_image, gpu::GpuMat& dst_image, const Point2f src_tri[], const Point2f dst_tri[]){
// 	if(dst_image.empty()) dst_image = src_image.clone();

	//アフィン変換行列を取得
	Mat aff_mat = getAffineTransform(src_tri, dst_tri);

	//アフィン変換
	gpu::GpuMat affed_image;
	warpAffine(gpu::GpuMat(src_image), affed_image, aff_mat, Size(dst_image.cols, dst_image.rows));


	vector<Point> mask_points(3);
	for(int i = 0; i < 3; i++){
		mask_points[i] = dst_tri[i];
	}

	//くり抜き用マスク
	Mat mask = Mat::zeros(dst_image.rows, dst_image.cols, dst_image.type());
	fillConvexPoly(mask, mask_points, Scalar(255,255,255));

	//くり抜き
	affed_image.copyTo(dst_image, gpu::GpuMat(mask));

// 	imshow("dst", dst_image);
// 	waitKey();
}
//三角形を取得してwarp
void App::warpGpu(const Mat& src_image, Mat& dst_image, const vector<Point2f>& src_points, const vector<Point2f>& dst_points){

	gpu::GpuMat src_image_gpu(src_image);
	gpu::GpuMat dst_image_gpu(dst_image);

	for(int i = 0; i < this->triangle_map.size(); i++){
		Point2f src_tri[3];
		Point2f dst_tri[3];
		for(int j = 0; j < 3; j++){
			src_tri[j] = src_points[ this->triangle_map[i][j] ];
			dst_tri[j] = dst_points[ this->triangle_map[i][j] ];
		}
		this->warpTriangleGpu(src_image_gpu, dst_image_gpu, src_tri, dst_tri);
	}

	dst_image = Mat(dst_image_gpu);
}

//形状から三角形idx群を生成
void App::makeTriangleIdxMap(const vector<Point2f>& src_points, vector<vector<int> >& tri_idxs){
	tri_idxs.clear();

	Rect rect_container = boundingRect(src_points);
	//境界上の点を含めるために+1する
	rect_container.width+=1;
	rect_container.height+=1;

	//平面分割
	Subdiv2D subdiv(rect_container);
	//点を登録
	subdiv.insert(src_points);

	//三角形を収納する変数
	vector<Vec6f> tris_6f;
	//三角形を取得
	subdiv.getTriangleList(tris_6f);

	//各三角形について
	for(vector<Vec6f>::iterator it = tris_6f.begin(); it != tris_6f.end(); it++){
		cv::Vec6f &vec = *it;

		bool skip_flag = false;
		//三角形の頂点idxを一時的に保存しておく変数
		vector<int> tmp_idxs(3);
		//三角形の各頂点について
		for(int i = 0; i < 3; i++){
			Point2f tmp_point(vec[2*i], vec[2*i+1]);
			//領域内の時、点を登録
			if(rect_container.contains(tmp_point)){
				//同一の点を検索
				int f;
				for(f = 0; f < src_points.size(); f++){
					if(src_points[f] == tmp_point) break;
				}
				tmp_idxs[i] = f;
			}
			//領域外の時、この三角形を無視
			else{
				skip_flag = true;
				break;
			}
		}
		//三角形の頂点を登録
		if(!skip_flag) tri_idxs.push_back(tmp_idxs);
	}
}
