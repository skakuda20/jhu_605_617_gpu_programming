#pragma once
#include <cuda_runtime.h>
#include <opencv2/core.hpp>

extern "C" {
void preprocess_frame_cuda(const cv::Mat& frame, cv::Mat& grayF, int w, int h, int stride, cudaStream_t stream);
}
