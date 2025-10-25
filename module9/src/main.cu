// noise_fft.cu

#define NO_CUDA_MATH_OVERLOADS
#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>

#include <cstdio>

#include "kernels.cuh"
#include "utils.h"

int main() {
    const int width = 256;
    const int height = 256;
    const int imgSize = width * height;
    const int freqSize = height * (width/2 + 1);

    printf("Creating %dx%d noise image...\n", width, height);

    // Allocate image memory
    thrust::device_vector<float> d_image(imgSize);
    float* d_image_ptr = thrust::raw_pointer_cast(d_image.data());

    // Generate Gaussian noise image (mean=0, stddev=1)
    generateGaussianNoise(d_image_ptr, imgSize);

    // FFT setup
    cufftHandle plan;
    cufftComplex* d_freqData;
    CHECK_CUDA(cudaMalloc(&d_freqData, sizeof(cufftComplex) * freqSize));

    // cuFFT plan R2C
    plan = createFFTPlan(width, height);

    // Forward FFT
    printf("Running forward FFT...\n");
    runFFT(plan, d_image_ptr, d_freqData);

    float* h_input = (float*)malloc(sizeof(float) * imgSize);
    CHECK_CUDA(cudaMemcpy(h_input, d_image_ptr, sizeof(float) * imgSize, cudaMemcpyDeviceToHost));
    printf("Stats BEFORE filtering:\n");
    statsOnly(h_input, imgSize);
    free(h_input);

    // Apply Gaussian filter in frequency domain
    dim3 block(16, 16);
    dim3 grid((width/2 + 1 + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    printf("Applying Gaussian low-pass filter...\n");
    applyGaussianFilter<<<grid, block>>>(d_freqData, width/2 + 1, height, 40.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Inverse FFT 
    cufftHandle planInverse;
    cufftPlan2d(&planInverse, width, height, CUFFT_C2R);
    printf("Running inverse FFT...\n");
    runIFFT(planInverse, d_freqData, d_image_ptr);

    // Check for CUDA errors before cuFFT plan
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error before cuFFT plan: %s\n", cudaGetErrorString(err));
    }

    // Copy result back to host
    float* h_output = (float*)malloc(sizeof(float) * imgSize);
    CHECK_CUDA(cudaMemcpy(h_output, d_image_ptr, sizeof(float) * imgSize, cudaMemcpyDeviceToHost));

    // Normalize result and output stats
    normalizeAndStats(h_output, imgSize);

    // Cleanup
    free(h_output);
    cufftDestroy(plan);
    cudaFree(d_freqData);

    printf("Done.\n");
    return 0;
}
