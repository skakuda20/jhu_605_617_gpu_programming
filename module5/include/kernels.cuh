#pragma once

#include <cuda_runtime.h>

// Kernel Configuration Constants
#define BLOCK_SIZE 16   // You can tune this (16x16 threads per block)

// Device Kernel Prototypes

// Erosion kernel (CUDA global function)
__global__ void erosionKernel(
    int* d_input,  // device input image
    int* d_output,       // device output image
    int width,                     // image width
    int height,                    // image height
    int radius               // structuring element radius (e.g., 1 for 3x3)
);


// Utility function to load halo pixels into shared memory
__device__ void load_halo_pixels(int tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2], const int* d_input,
                                 int globalX, int globalY, int width, int height, int radius);

void copy_to_constant_memory(const int* hostMask);

// Constant Memory Declaration
// Structuring element (e.g., 3x3 mask)
// Filled from host with cudaMemcpyToSymbol()
extern __constant__ int d_structuringElement[9];