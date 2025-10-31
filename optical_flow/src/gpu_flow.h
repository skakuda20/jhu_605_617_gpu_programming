
#pragma once
#include <opencv2/opencv.hpp>

// GPU wrapper: compute optical flow on GPU (Horn-Schunck style)
// Expects input single-channel CV_32F images
void hornSchunckGPU(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &flow,
                    float alpha=1.0f, int iterations=100, cudaStream_t stream = 0);
