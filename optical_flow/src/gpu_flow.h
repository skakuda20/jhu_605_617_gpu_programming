
#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// GPU wrapper for computing optical flow on GPU using the Horn-Schunck method
void hornSchunckGPU(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &flow,
                    float alpha=1.0f, int iterations=100, cudaStream_t stream = 0);
