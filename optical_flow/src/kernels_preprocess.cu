// kernels_preprocess.cu
#include "kernels_preprocess.cuh"

// Grayscale conversion (BGR to Gray)
__global__ void bgr2gray_kernel(const uchar3* input, float* output, int w, int h, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * stride + x;
    uchar3 pix = input[idx];
    output[idx] = 0.299f * pix.x + 0.587f * pix.y + 0.114f * pix.z;
}

// Histogram equalization (global, naive)
__global__ void hist_eq_kernel(const float* input, float* output, int w, int h, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * stride + x;
    output[idx] = input[idx]; // No-op
}

// Laplacian edge enhancement
__global__ void laplacian_kernel(const float* input, float* output, int w, int h, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= w-1 || y >= h-1) return;
    int idx = y * stride + x;
    float center = input[idx];
    float sum = input[idx-stride] + input[idx+stride] + input[idx-1] + input[idx+1] - 4.0f * center;
    output[idx] = center + 0.2f * sum; // Weighted Laplacian
}

// Min/max reduction for contrast stretching
__global__ void minmax_kernel(const float* input, float* minOut, float* maxOut, int w, int h, int stride) {
    // For brevity, this is a placeholder. Real implementation would use reduction.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *minOut = 0.0f;
        *maxOut = 1.0f;
    }
}

// Contrast stretching
__global__ void contrast_stretch_kernel(const float* input, float* output, float minVal, float maxVal, int w, int h, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * stride + x;
    output[idx] = (input[idx] - minVal) / (maxVal - minVal);
}

// Gaussian blur (naive, box filter)
__global__ void gaussian_blur_kernel(const float* input, float* output, int w, int h, int stride, int ksize, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * stride + x;
    float sum = 0.0f;
    int count = 0;
    for (int dy = -ksize/2; dy <= ksize/2; ++dy) {
        for (int dx = -ksize/2; dx <= ksize/2; ++dx) {
            int nx = min(max(x+dx,0),w-1);
            int ny = min(max(y+dy,0),h-1);
            sum += input[ny*stride + nx];
            count++;
        }
    }
    output[idx] = sum / count;
}

// Convert to float and normalize
__global__ void normalize_kernel(const unsigned char* input, float* output, int w, int h, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * stride + x;
    output[idx] = input[idx] / 255.0f;
}
