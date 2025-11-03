#include "gpu_flow.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

void hornSchunckGPU(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &flow,
                    float alpha, int iterations, cudaStream_t stream) {
    CV_Assert(I1.type() == CV_32F && I2.type() == CV_32F);
    int rows = I1.rows, cols = I1.cols;
    int stride = cols; // assuming no padding

    // Allocate device memory
    size_t imgBytes = rows * stride * sizeof(float);
    float *d_I1, *d_I2, *d_Ix, *d_Iy, *d_It, *d_u, *d_v, *d_uNew, *d_vNew;
    cudaMalloc(&d_I1, imgBytes);
    cudaMalloc(&d_I2, imgBytes);
    cudaMalloc(&d_Ix, imgBytes);
    cudaMalloc(&d_Iy, imgBytes);
    cudaMalloc(&d_It, imgBytes);
    cudaMalloc(&d_u, imgBytes);
    cudaMalloc(&d_v, imgBytes);
    cudaMalloc(&d_uNew, imgBytes);
    cudaMalloc(&d_vNew, imgBytes);

    // Copy input images to device
    cudaMemcpyAsync(d_I1, I1.ptr<float>(), imgBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_I2, I2.ptr<float>(), imgBytes, cudaMemcpyHostToDevice, stream);

    // Initialize u, v to zero
    cudaMemsetAsync(d_u, 0, imgBytes, stream);
    cudaMemsetAsync(d_v, 0, imgBytes, stream);

    // Launch gradient kernel
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    compute_gradients_kernel<<<grid, block, 0, stream>>>(d_I1, d_I2, d_Ix, d_Iy, d_It, cols, rows, stride);

    float alpha2 = alpha * alpha;

    // Iterative update
    for (int iter = 0; iter < iterations; ++iter) {
        hs_iteration_kernel<<<grid, block, 0, stream>>>(d_Ix, d_Iy, d_It,
                                                        d_u, d_v, d_uNew, d_vNew,
                                                        cols, rows, stride, alpha2);
        // Swap pointers for next iteration
        std::swap(d_u, d_uNew);
        std::swap(d_v, d_vNew);
    }

    // Copy results back to host
    std::vector<float> h_u(rows * stride), h_v(rows * stride);
    cudaMemcpyAsync(h_u.data(), d_u, imgBytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_v.data(), d_v, imgBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Pack into flow matrix
    flow.create(rows, cols, CV_32FC2);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int idx = y * stride + x;
            flow.at<cv::Point2f>(y, x) = cv::Point2f(h_u[idx], h_v[idx]);
        }
    }

    // Free device memory
    cudaFree(d_I1);
    cudaFree(d_I2);
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_It);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_uNew);
    cudaFree(d_vNew);
}