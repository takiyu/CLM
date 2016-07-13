#include "fps.h"

using namespace cv;

FpsCounter::FpsCounter() { this->init(); }

void FpsCounter::init() {
    frame_count = 0;
    pre_frame_count = 0;
    now_time = 0;
    time_diff = 0;
    fps = 0;
    frequency = (1000.0 / cv::getTickFrequency());
    start_time = cv::getTickCount();
}

void FpsCounter::updateFrame() {
    now_time = cv::getTickCount();
    time_diff = (int)((now_time - start_time) * frequency);
    if (time_diff >= 1000) {
        start_time = now_time;
        fps = frame_count - pre_frame_count;
        pre_frame_count = frame_count;
    }
    frame_count++;
}
