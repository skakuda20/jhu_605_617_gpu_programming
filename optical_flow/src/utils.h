#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <iostream>
#else
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ", code " << err << " (" << cudaGetErrorString(err) << ")\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA kernel launch error at " << __FILE__ << ":" << __LINE__ << ", code " << err << " (" << cudaGetErrorString(err) << ")\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

    #define CUDA_CHECK_SYNC() \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA sync error at " << __FILE__ << ":" << __LINE__ << ", code " << err << " (" << cudaGetErrorString(err) << ")\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

    #endif
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

// Draw flow vectors 
void drawFlowArrows(cv::Mat &img, const cv::Mat &flow, int step=16, const cv::Scalar &color = cv::Scalar(0,255,0));
cv::Mat flowToColor(const cv::Mat &flow);
