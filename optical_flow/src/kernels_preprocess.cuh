// kernels_preprocess.cuh
#pragma once
#include <cuda_runtime.h>

extern "C" {
// Grayscale conversion (BGR to Gray)
__global__ void bgr2gray_kernel(const uchar3* input, float* output, int w, int h, int stride);

// Histogram equalization (global)
__global__ void hist_eq_kernel(const float* input, float* output, int w, int h, int stride);

// Laplacian edge enhancement
__global__ void laplacian_kernel(const float* input, float* output, int w, int h, int stride);

// Min/max reduction for contrast stretching
__global__ void minmax_kernel(const float* input, float* minOut, float* maxOut, int w, int h, int stride);

// Contrast stretching
__global__ void contrast_stretch_kernel(const float* input, float* output, float minVal, float maxVal, int w, int h, int stride);

// Gaussian blur
__global__ void gaussian_blur_kernel(const float* input, float* output, int w, int h, int stride, int ksize, float sigma);

// Convert to float and normalize
__global__ void normalize_kernel(const unsigned char* input, float* output, int w, int h, int stride);
}
