// noise_fft.cu

#include "cuda_compat.h"

#define NO_CUDA_MATH_OVERLOADS
#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>

#include "kernels.cuh"
#include "utils.h"

#include <cstdio>
// #include <math.h>

// cudaSetDevice(0);

int main() {
    const int width = 256;
    const int height = 256;
    const int imgSize = width * height;
    const int freqSize = height * (width/2 + 1);

    printf("Creating %dx%d noise image...\n", width, height);

    // Allocate image memory
    float* d_image;
    CHECK_CUDA(cudaMalloc(&d_image, sizeof(float) * imgSize));

    // Generate Gaussian noise image (mean=0, stddev=1)
    generateGaussianNoise(d_image, imgSize);

    // FFT setup
    cufftHandle plan;
    cufftComplex* d_freqData;
    CHECK_CUDA(cudaMalloc(&d_freqData, sizeof(cufftComplex) * freqSize));

    // cuFFT plan: R2C (real-to-complex) 2D
    plan = createFFTPlan(width, height);

    // Execute forward FFT
    printf("Running forward FFT...\n");
    runFFT(plan, d_image, d_freqData);

    float* h_input = (float*)malloc(sizeof(float) * imgSize);
    CHECK_CUDA(cudaMemcpy(h_input, d_image, sizeof(float) * imgSize, cudaMemcpyDeviceToHost));
    printf("Stats BEFORE filtering:\n");
    statsOnly(h_input, imgSize);
    free(h_input);

    // Apply Gaussian filter in frequency domain
    dim3 block(16, 16);
    dim3 grid((width/2 + 1 + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    printf("Applying Gaussian low-pass filter...\n");
    applyGaussianFilter<<<grid, block>>>(d_freqData, width/2 + 1, height, 40.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Execute inverse FFT (complex-to-real)
    cufftHandle planInverse;
    cufftPlan2d(&planInverse, width, height, CUFFT_C2R);
    printf("Running inverse FFT...\n");
    runIFFT(planInverse, d_freqData, d_image);

    // Check for CUDA errors before cuFFT plan
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error before cuFFT plan: %s\n", cudaGetErrorString(err));
    }

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
