#pragma once

#include <cuda_runtime.h>

// Device Kernel Prototypes

// Erosion kernel (CUDA global function)
__global__ void erosionKernel(
    int* d_input,        // device input image
    int* d_output,       // device output image
    int width,           // image width
    int height,          // image height
    int radius           // structuring element radius
);


// Utility function to load halo pixels into shared memory
__device__ void load_halo_pixels(int* tile, int tilePitch, const int* d_input,
                                 int globalX, int globalY, int width, int height, int radius, int blockDimX, int blockDimY);

void copy_to_constant_memory(const int* hostMask);

// Constant Memory Declaration
extern __constant__ int d_structuringElement[9];