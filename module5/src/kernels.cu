// kernels.cu
#include "kernels.cuh"

// Constant memory for structuring element
__constant__ int d_structuringElement[9];


__device__ void load_halo_pixels(int tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2], const int* d_input,
                                 int globalX, int globalY, int width, int height, int radius) {
    // Load halo pixels around the block (edges and corners)
    // Each thread loads its own halo pixel if at the border
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int lx = threadIdx.x + radius + dx;
            int ly = threadIdx.y + radius + dy;
            int gx = globalX + dx;
            int gy = globalY + dy;
            if (lx >= 0 && lx < BLOCK_SIZE + 2 && ly >= 0 && ly < BLOCK_SIZE + 2 &&
                gx >= 0 && gx < width && gy >= 0 && gy < height) {
                tile[ly][lx] = d_input[gy * width + gx];
            }
        }
    }
}


__global__ void erosionKernel(int* d_input, int* d_output, int width, int height, int radius) {
    // Shared memory tile with halo
    __shared__ int tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Compute global coords
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;
    int localX = threadIdx.x + radius;
    int localY = threadIdx.y + radius;

    // Load central pixel into shared memory
    if (globalX < width && globalY < height) {
        tile[localY][localX] = d_input[globalY * width + globalX];
    }

    // Load halo pixels (neighbors at block edges)
    load_halo_pixels(tile, d_input, globalX, globalY, width, height, radius);

    __syncthreads();

    // Apply erosion if within image bounds
    if (globalX < width && globalY < height) {
        int minVal = 255;   // register

        // Loop through structuring element
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                if (d_structuringElement[(ky + 1) * 3 + (kx + 1)] == 1) {
                    int neighbor = tile[localY + ky][localX + kx];
                    minVal = min(minVal, neighbor);
                }
            }
        }

        // Write result to global memory
        d_output[globalY * width + globalX] = minVal;
    }