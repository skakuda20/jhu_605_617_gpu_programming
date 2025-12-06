#include "pipeline.cuh"
#include "kernels_preprocess.cuh"
#include <cuda_runtime.h>
#include <opencv2/core.hpp>


extern "C" void preprocess_frame_cuda(const cv::Mat& frame, cv::Mat& grayF, int w, int h, int stride, cudaStream_t stream) {
    // Allocation and initialization
    static uchar3* d_frame = nullptr;
    static float *d_gray = nullptr, *d_eq = nullptr, *d_edges = nullptr, *d_blur = nullptr;
    static float *d_min = nullptr, *d_max = nullptr;

    // Allocation and initialization
    if (!d_frame) {
        cudaMalloc(&d_frame, w * h * sizeof(uchar3));
        cudaMalloc(&d_gray, w * h * sizeof(float));
        cudaMalloc(&d_eq, w * h * sizeof(float));
        cudaMalloc(&d_edges, w * h * sizeof(float));
        cudaMalloc(&d_blur, w * h * sizeof(float));
        cudaMalloc(&d_min, sizeof(float));
        cudaMalloc(&d_max, sizeof(float));

        cudaMemsetAsync(d_gray, 0, w * h * sizeof(float), stream);
        cudaMemsetAsync(d_eq, 0, w * h * sizeof(float), stream);
        cudaMemsetAsync(d_edges, 0, w * h * sizeof(float), stream);
        cudaMemsetAsync(d_blur, 0, w * h * sizeof(float), stream);
        cudaMemsetAsync(d_min, 0, sizeof(float), stream);
        cudaMemsetAsync(d_max, 0, sizeof(float), stream);
    }

    // Error checking helper
    auto checkCuda = [](const char* msg) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[CUDA ERROR] %s: %s\n", msg, cudaGetErrorString(err));
        }
    };

    // Copy input frame to device
    cudaMemcpyAsync(d_frame, frame.ptr<uchar3>(), w * h * sizeof(uchar3), cudaMemcpyHostToDevice, stream);
    checkCuda("MemcpyAsync d_frame");

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    // Convert BGR to grayscale
    bgr2gray_kernel<<<grid, block, 0, stream>>>(d_frame, d_gray, w, h, stride);
    checkCuda("bgr2gray_kernel");

    std::vector<float> h_gray(w * h);
    cudaMemcpyAsync(h_gray.data(), d_gray, w * h * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float sum_gray = 0.0f;
    for (size_t i = 0; i < h_gray.size(); ++i) sum_gray += h_gray[i];
    printf("[DIAG] d_gray[0]=%f, [h/2*w/2]=%f, mean=%f\n", 
           h_gray[0], h_gray[(h/2) * stride + (w/2)], sum_gray / (w * h));

    // Histogram equalization
    hist_eq_kernel<<<grid, block, 0, stream>>>(d_gray, d_eq, w, h, stride);
    checkCuda("hist_eq_kernel");

    std::vector<float> h_eq(w * h);
    cudaMemcpyAsync(h_eq.data(), d_eq, w * h * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float sum_eq = 0.0f;
    for (size_t i = 0; i < h_eq.size(); ++i) sum_eq += h_eq[i];
    printf("[DIAG] d_eq[0]=%f, [h/2*w/2]=%f, mean=%f\n", 
           h_eq[0], h_eq[(h/2) * stride + (w/2)], sum_eq / (w * h));

    // Edge detection
    laplacian_kernel<<<grid, block, 0, stream>>>(d_eq, d_edges, w, h, stride);
    checkCuda("laplacian_kernel");

    std::vector<float> h_edges(w * h);
    cudaMemcpyAsync(h_edges.data(), d_edges, w * h * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float sum_edges = 0.0f;
    for (size_t i = 0; i < h_edges.size(); ++i) sum_edges += h_edges[i];
    printf("[DIAG] d_edges[0]=%f, [h/2*w/2]=%f, mean=%f\n", 
           h_edges[0], h_edges[(h/2) * stride + (w/2)], sum_edges / (w * h));

    // Blend equalized image with edges
    cv::Mat eq(h, w, CV_32F), edges(h, w, CV_32F);
    cudaMemcpyAsync(eq.ptr<float>(), d_eq, w * h * sizeof(float), cudaMemcpyDeviceToHost, stream);
    checkCuda("MemcpyAsync d_eq to eq");
    cudaMemcpyAsync(edges.ptr<float>(), d_edges, w * h * sizeof(float), cudaMemcpyDeviceToHost, stream);
    checkCuda("MemcpyAsync d_edges to edges");
    cudaStreamSynchronize(stream);

    cv::addWeighted(eq, 0.8, edges, 0.2, 0, eq);
    cudaMemcpyAsync(d_eq, eq.ptr<float>(), w * h * sizeof(float), cudaMemcpyHostToDevice, stream);
    checkCuda("MemcpyAsync eq to d_eq");

    // Contrast stretching
    auto minmax = std::minmax_element(h_eq.begin(), h_eq.end());
    float minVal = *minmax.first;
    float maxVal = *minmax.second;
    printf("[HOST] minVal: %f, maxVal: %f\n", minVal, maxVal);

    if (fabs(maxVal - minVal) < 1e-6) {
        cudaMemsetAsync(d_eq, 0, w * h * sizeof(float), stream);
        checkCuda("MemsetAsync d_eq");
    } else {
        contrast_stretch_kernel<<<grid, block, 0, stream>>>(d_eq, d_eq, minVal, maxVal, w, h, stride);
        checkCuda("contrast_stretch_kernel");

        std::vector<float> h_eq_stretch(w * h);
        cudaMemcpyAsync(h_eq_stretch.data(), d_eq, w * h * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        float sum_eq_stretch = 0.0f;
        for (size_t i = 0; i < h_eq_stretch.size(); ++i) sum_eq_stretch += h_eq_stretch[i];
        printf("[DIAG] d_eq_stretch[0]=%f, [h/2*w/2]=%f, mean=%f\n", 
               h_eq_stretch[0], h_eq_stretch[(h/2) * stride + (w/2)], sum_eq_stretch / (w * h));
    }

    // Gaussian blur
    gaussian_blur_kernel<<<grid, block, 0, stream>>>(d_eq, d_blur, w, h, stride, 3, 0.0f);
    checkCuda("gaussian_blur_kernel");

    std::vector<float> h_blur(w * h);
    cudaMemcpyAsync(h_blur.data(), d_blur, w * h * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float sum_blur = 0.0f;
    for (size_t i = 0; i < h_blur.size(); ++i) sum_blur += h_blur[i];
    printf("[DIAG] d_blur[0]=%f, [h/2*w/2]=%f, mean=%f\n", 
           h_blur[0], h_blur[(h/2) * stride + (w/2)], sum_blur / (w * h));

    grayF.create(h, w, CV_32F);
    cudaMemcpyAsync(grayF.ptr<float>(), d_blur, w * h * sizeof(float), cudaMemcpyDeviceToHost, stream);
    checkCuda("MemcpyAsync d_blur to grayF");
    cudaStreamSynchronize(stream);
}
