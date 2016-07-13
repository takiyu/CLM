#ifndef FPSCOUNTER_H_141021
#define FPSCOUNTER_H_141021

#include <opencv2/core/core.hpp>

class FpsCounter {
public:
    FpsCounter();
    void init();

    // Call for each frame
    void updateFrame();

    // Get current fps
    int getFps() { return fps; }

private:
    int frame_count, pre_frame_count;
    int64 now_time;
    int64 time_diff;
    int fps;
    double frequency;
    int64 start_time;
};

#endif
