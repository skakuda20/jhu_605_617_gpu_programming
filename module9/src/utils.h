// utils.h

#pragma once

#define NO_CUDA_MATH_OVERLOADS
#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>
#include <cstdio>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>


// Define CUDA checks
#define CHECK_CUDA(call) \
    { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); exit(EXIT_FAILURE); }}

#define CHECK_CURAND(call) \
    { curandStatus_t status = (call); if (status != CURAND_STATUS_SUCCESS) { \
    fprintf(stderr, "cuRAND Error at %s:%d\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }}

#define CHECK_CUFFT(call) \
    { cufftResult res = (call); if (res != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT Error at %s:%d (code=%d)\n", __FILE__, __LINE__, res); exit(EXIT_FAILURE); }}


// Helper functions for noise generation and FFT operations
void generateGaussianNoise(float* d_image, int imgSize, unsigned long long seed = 1234ULL) {
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CHECK_CURAND(curandGenerateNormal(gen, d_image, imgSize, 0.0f, 1.0f));
    curandDestroyGenerator(gen);
}

cufftHandle createFFTPlan(int width, int height) {
    cufftHandle plan;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error before cuFFT plan: %s\n", cudaGetErrorString(err));
    }
    CHECK_CUFFT(cufftPlan2d(&plan, width, height, CUFFT_R2C));
    return plan;
}

void runFFT(cufftHandle plan, float* d_image, cufftComplex* d_freqData) {
    CHECK_CUFFT(cufftExecR2C(plan, d_image, d_freqData));
}

void runIFFT(cufftHandle plan, cufftComplex* d_freqData, float* d_image) {
    CHECK_CUFFT(cufftExecC2R(plan, d_freqData, d_image));
}

void statsOnly(float* h_output, int imgSize) {
    float minVal = 1e9, maxVal = -1e9;
    for (int i = 0; i < imgSize; ++i) {
        minVal = fminf(minVal, h_output[i]);
        maxVal = fmaxf(maxVal, h_output[i]);
    }
    printf("Raw image stats: min=%.4f max=%.4f\n", minVal, maxVal);
}

void normalizeAndStats(float* h_output, int imgSize) {
    for (int i = 0; i < imgSize; ++i) h_output[i] /= imgSize;
    float minVal = 1e9, maxVal = -1e9, sum = 0.0f;
    for (int i = 0; i < imgSize; ++i) {
        minVal = fminf(minVal, h_output[i]);
        maxVal = fmaxf(maxVal, h_output[i]);
        sum += h_output[i];
    }
    float mean = sum / imgSize;
    printf("Filtered image stats: min=%.4f max=%.4f mean=%.4f\n", minVal, maxVal, mean);
}

void normalizeAndStats(thrust::device_vector<float>& d_image, int imgSize) {
    float* d_image_ptr = thrust::raw_pointer_cast(d_image.data());
    thrust::transform(d_image_ptr, d_image_ptr + imgSize, d_image_ptr,
                 [imgSize] __device__ (float x) { return x / imgSize; });
    float minVal = *thrust::min_element(d_image_ptr, d_image_ptr + imgSize);
    float maxVal = *thrust::max_element(d_image_ptr, d_image_ptr + imgSize);
    float sum = thrust::reduce(d_image_ptr, d_image_ptr + imgSize, 0.0f, thrust::plus<float>());
    float mean = sum / imgSize;
    printf("Filtered image stats: min=%.4f max=%.4f mean=%.4f\n", minVal, maxVal, mean);
}