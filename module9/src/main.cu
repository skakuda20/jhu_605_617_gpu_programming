// noise_fft.cu
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>

#inlcude "kernels.cuh"
#include "utils.h"



int main() {
    const int width = 512;
    const int height = 512;
    const int imgSize = width * height;

    printf("Creating %dx%d noise image...\n", width, height);

    // Allocate image memory
    float* d_image;
    CHECK_CUDA(cudaMalloc(&d_image, sizeof(float) * imgSize));

    // Generate Gaussian noise image (mean=0, stddev=1)
    generateGaussianNoise(d_image, imgSize);

    // FFT setup
    cufftHandle plan;
    cufftComplex* d_freqData;
    CHECK_CUDA(cudaMalloc(&d_freqData, sizeof(cufftComplex) * imgSize));

    // cuFFT plan: R2C (real-to-complex) 2D
    plan = createFFTPlan(width, height);

    // Execute forward FFT
    printf("Running forward FFT...\n");
    runFFT(plan, d_image, d_freqData);

    // Apply Gaussian filter in frequency domain
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    printf("Applying Gaussian low-pass filter...\n");
    applyGaussianFilter<<<grid, block>>>(d_freqData, width, height, 40.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Execute inverse FFT (complex-to-real)
    printf("Running inverse FFT...\n");
    runIFFT(plan, d_freqData, d_image);

    // Copy result back to host
    float* h_output = (float*)malloc(sizeof(float) * imgSize);
    CHECK_CUDA(cudaMemcpy(h_output, d_image, sizeof(float) * imgSize, cudaMemcpyDeviceToHost));

    // Normalize result and output stats
    normalizeAndStats(h_output, imgSize);

    // Cleanup
    free(h_output);
    cufftDestroy(plan);
    cudaFree(d_image);
    cudaFree(d_freqData);

    printf("Done.\n");
    return 0;
}
