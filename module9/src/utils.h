// utils.h

#pragma once
#include <cuda_runtime.h>
#include <iostream>


#define CHECK_CUDA(call) \
    { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); exit(EXIT_FAILURE); }}

#define CHECK_CURAND(call) \
    { curandStatus_t status = (call); if (status != CURAND_STATUS_SUCCESS) { \
    fprintf(stderr, "cuRAND Error at %s:%d\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }}

#define CHECK_CUFFT(call) \
    { cufftResult res = (call); if (res != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT Error at %s:%d\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }}


// =======================================================================================

void generateGaussianNoise(float* d_image, int imgSize, unsigned long long seed = 1234ULL) {
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CHECK_CURAND(curandGenerateNormal(gen, d_image, imgSize, 0.0f, 1.0f));
    curandDestroyGenerator(gen);
}

cufftHandle createFFTPlan(int width, int height) {
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan2d(&plan, height, width, CUFFT_R2C));
    return plan;
}

void runFFT(cufftHandle plan, float* d_image, cufftComplex* d_freqData) {
    CHECK_CUFFT(cufftExecR2C(plan, d_image, d_freqData));
}

void runIFFT(cufftHandle plan, cufftComplex* d_freqData, float* d_image) {
    CHECK_CUFFT(cufftExecC2R(plan, d_freqData, d_image));
}

void normalizeAndStats(float* h_output, int imgSize) {
    for (int i = 0; i < imgSize; ++i) h_output[i] /= imgSize;
    float minVal = 1e9, maxVal = -1e9;
    for (int i = 0; i < imgSize; ++i) {
        minVal = fminf(minVal, h_output[i]);
        maxVal = fmaxf(maxVal, h_output[i]);
    }
    printf("Filtered image stats: min=%.4f max=%.4f\n", minVal, maxVal);
}