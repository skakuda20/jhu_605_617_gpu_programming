#include <cufft.h>

// GPU kernel to apply a Gaussian low-pass filter in the frequency domain
__global__ void applyGaussianFilter(cufftComplex* freqData, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Compute distance from center
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float dx = (x - cx) / sigma;
    float dy = (y - cy) / sigma;
    float gauss = expf(-(dx * dx + dy * dy) / 2.0f);

    freqData[idx].x *= gauss;
    freqData[idx].y *= gauss;
}