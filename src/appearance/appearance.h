#ifndef APPEARANCE_H_141129
#define APPEARANCE_H_141129

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

class App{
public:
	//コンストラクタ
	App();
	//初期化付きコンストラクタ
	App(const vector<Point2f> init_shape);
	//平面分割の初期化
	void init(const vector<Point2f> init_shape);

	//三角形でwarp
	void warpTriangle(const Mat& src_image, Mat& dst_image, const Point2f src_tri[], const Point2f dst_tri[]);
	//三角形を取得してwarp
	void warp(const Mat& src_im, Mat& dst_im, const vector<Point2f>& src_points, const vector<Point2f>& dst_points);

	//三角形でwarp(gpu)
	void warpTriangleGpu(const gpu::GpuMat& src_image, gpu::GpuMat& dst_image, const Point2f src_tri[], const Point2f dst_tri[]);
	//三角形を取得してwarp(gpu)
	void warpGpu(const Mat& src_im, Mat& dst_im, const vector<Point2f>& src_points, const vector<Point2f>& dst_points);


private:
	//三角形を生成するMap
	vector<vector<int> > triangle_map;

	//形状から三角形idx群を生成
	void makeTriangleIdxMap(const vector<Point2f>& src_points, vector<vector<int> >& tri_idxs);
};

#endif
