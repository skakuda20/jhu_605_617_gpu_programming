#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

// Kernel Configuration Constants
#define BLOCK_SIZE 16   // You can tune this (16x16 threads per block)

// Device Kernel Prototypes

// Erosion kernel (CUDA global function)
__global__ void erosionKernel(
    const unsigned char* d_input,  // device input image
    unsigned char* d_output,       // device output image
    int width,                     // image width
    int height,                    // image height
    int kernelRadius               // structuring element radius (e.g., 1 for 3x3)
);

// Constant Memory Declaration
// Structuring element (e.g., 3x3 mask)
// Filled from host with cudaMemcpyToSymbol()
__constant__ int d_structuringElement[9];

#endif // KERNELS_CUH
