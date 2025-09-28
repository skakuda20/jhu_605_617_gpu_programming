// kernels.cu
#include "kernels.cuh"

__device__ void load_halo_pixels(int* tile, int tilePitch, const int* d_input,
                                 int globalX, int globalY, int width, int height, int radius, int blockDimX, int blockDimY) {
    int tileX = blockDimX + 2 * radius;
    int tileY = blockDimY + 2 * radius;
    // Initialize all shared memory to 255 (max value for erosion)
    for (int y = threadIdx.y; y < tileY; y += blockDim.y) {
        for (int x = threadIdx.x; x < tileX; x += blockDim.x) {
            tile[y * tilePitch + x] = 255;
        }
    }
    __syncthreads();

    // Load halo and central pixels
    for (int dy = -radius; dy <= blockDimY + radius - 1; ++dy) {
        for (int dx = -radius; dx <= blockDimX + radius - 1; ++dx) {
            int lx = threadIdx.x + radius + dx;
            int ly = threadIdx.y + radius + dy;
            int gx = globalX + dx;
            int gy = globalY + dy;
            if (lx >= 0 && lx < tileX && ly >= 0 && ly < tileY) {
                if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                    tile[ly * tilePitch + lx] = d_input[gy * width + gx];
                } else {
                    tile[ly * tilePitch + lx] = 255;
                }
            }
        }
    }
    __syncthreads();
}


// Host function to copy mask to constant memory
void copy_to_constant_memory(const int* hostMask) {
    cudaMemcpyToSymbol(d_structuringElement, hostMask, 9 * sizeof(int));
}

__global__ void erosionKernel(int* d_input, int* d_output, int width, int height, int radius) {
    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    int tilePitch = blockDimX + 2 * radius;
    int tileHeight = blockDimY + 2 * radius;
    extern __shared__ int tile[];

    // Compute global coords
    int globalX = blockIdx.x * blockDimX + threadIdx.x;
    int globalY = blockIdx.y * blockDimY + threadIdx.y;
    int localX = threadIdx.x + radius;
    int localY = threadIdx.y + radius;

    // Load halo pixels (neighbors at block edges)
    load_halo_pixels(tile, tilePitch, d_input, globalX, globalY, width, height, radius, blockDimX, blockDimY);

    __syncthreads();

    // Apply erosion if within image bounds
    if (globalX < width && globalY < height) {
        int minVal = 255;   // register

        // Loop through structuring element
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                if (d_structuringElement[(ky + 1) * 3 + (kx + 1)] == 1) {
                    int neighbor = tile[(localY + ky) * tilePitch + (localX + kx)];
                    minVal = min(minVal, neighbor);
                }
            }
        }

        // Write result to global memory
        d_output[globalY * width + globalX] = minVal;
    }
}

__constant__ int d_structuringElement[9];