#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

struct Timer {
    std::chrono::time_point<Clock> start;
    void tic(){ start = Clock::now(); }
    double toc_ms(){
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// draw flow vectors (sparse sampling)
void drawFlowArrows(cv::Mat &img, const cv::Mat &flow, int step=16, const cv::Scalar &color = cv::Scalar(0,255,0));
cv::Mat flowToColor(const cv::Mat &flow); // produce a visualization for flow (HSV)
