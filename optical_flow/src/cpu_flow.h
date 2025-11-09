#pragma once
#include <opencv2/opencv.hpp>

// Compute dense Horn-Schunck optical flow between two grayscale images
void hornSchunckCPU(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &flow, float alpha=1.0f, int iterations=100);
