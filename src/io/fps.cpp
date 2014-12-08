#include "fps.h"

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

//======FpsCounter======
//コンストラクタ
FpsCounter::FpsCounter(){
	this->init();
}
//初期化
void FpsCounter::init(){
	frame_count = 0;    // frame数
	pre_frame_count = 0;    // 前フレーム数
	now_time = 0;   // 現時刻
	time_diff = 0;   // 経過時間
	fps = 0;    // 1秒のフレーム数
	frequency = (1000 /cv::getTickFrequency());
	start_time = cv::getTickCount();
}
//１フレームに１回呼ぶ
void FpsCounter::updateFrame(){
	now_time = cv::getTickCount();  
	time_diff = (int)((now_time - start_time) * frequency);
	if (time_diff >= 1000) {
		start_time = now_time;
		fps = frame_count - pre_frame_count;
		pre_frame_count = frame_count;
	}
	frame_count++;
}
