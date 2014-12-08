#ifndef FPSCOUNTER_H_141021
#define FPSCOUNTER_H_141021

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
class FpsCounter{
public:
	//コンストラクタ
	FpsCounter();
	//初期化
	void init();
	//１フレームに１回呼ぶ
	void updateFrame();
	//fpsを取得
	int getFps(){ return fps; }
private:
	int frame_count;		// frame数
	int pre_frame_count;	// 前フレーム数
	int64 now_time;			// 現時刻
	int64 time_diff;		// 経過時間
	int fps;
	double frequency;		//フレームの更新期間
	int64 start_time;
};

#endif
