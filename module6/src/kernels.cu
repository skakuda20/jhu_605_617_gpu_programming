#include "kernels.cuh"

// load_halo_pixels implementation (already provided in kernels.cuh, can be left as is or moved here if needed)
__device__ void load_halo_pixels(int* tile, int tilePitch, const int* d_input,
                                 int globalX, int globalY, int width, int height, int radius, int blockDimX, int blockDimY) {
    int tileX = blockDimX + 2 * radius;
    int tileY = blockDimY + 2 * radius;
    for (int y = threadIdx.y; y < tileY; y += blockDim.y) {
        for (int x = threadIdx.x; x < tileX; x += blockDim.x) {
            tile[y * tilePitch + x] = 255;
        }
    }
    __syncthreads();

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

// Erosion Kernel
__global__ void erosionKernel(int* d_input, int* d_output, int width, int height, int radius) {
    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    int tilePitch = blockDimX + 2 * radius;
    int tileHeight = blockDimY + 2 * radius;
    extern __shared__ int tile[];

    int globalX = blockIdx.x * blockDimX + threadIdx.x;
    int globalY = blockIdx.y * blockDimY + threadIdx.y;
    int localX = threadIdx.x + radius;
    int localY = threadIdx.y + radius;

    load_halo_pixels(tile, tilePitch, d_input, globalX, globalY, width, height, radius, blockDimX, blockDimY);

    __syncthreads();

    if (globalX < width && globalY < height) {
        int minVal = 255;
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                if (d_structuringElement[(ky + 1) * 3 + (kx + 1)] == 1) {
                    int neighbor = tile[(localY + ky) * tilePitch + (localX + kx)];
                    minVal = min(minVal, neighbor);
                }
            }
        }
        d_output[globalY * width + globalX] = minVal;
    }
}

// Constant memory for structuring element
__constant__ int d_structuringElement[9];

// Simple inversion kernel
__global__ void invertKernel(int* d_input, int* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        d_output[idx] = 255 - d_input[idx];
    }
}

// Simple 3x3 box blur kernel using shared memory
__global__ void blurKernel(int* d_input, int* d_output, int width, int height) {
    const int radius = 1; // 3x3 kernel
    int tileX = blockDim.x + 2 * radius;
    int tileY = blockDim.y + 2 * radius;
    extern __shared__ int tile[];

    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;
    int localX = threadIdx.x + radius;
    int localY = threadIdx.y + radius;

    // Load halo pixels
    for (int y = threadIdx.y; y < tileY; y += blockDim.y) {
        for (int x = threadIdx.x; x < tileX; x += blockDim.x) {
            int gx = blockIdx.x * blockDim.x + x - radius;
            int gy = blockIdx.y * blockDim.y + y - radius;
            if (gx >= 0 && gx < width && gy >= 0 && gy < height)
                tile[y * tileX + x] = d_input[gy * width + gx];
            else
                tile[y * tileX + x] = 0;
        }
    }
    __syncthreads();

    // Apply 3x3 box blur
    if (globalX < width && globalY < height) {
        int sum = 0;
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                sum += tile[(localY + ky) * tileX + (localX + kx)];
            }
        }
        d_output[globalY * width + globalX] = sum / 9;
    }
}

// Example Gaussian blur kernel (simple version, not optimized)
__global__ void gaussianBlurKernel(int* d_input, int* d_output, int width, int height) {
    // 3x3 Gaussian kernel: [1 2 1; 2 4 2; 1 2 1] normalized by 16
    const int kernel[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int sum = 0;
        int weight = 0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int nx = x + kx;
                int ny = y + ky;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += d_input[ny * width + nx] * kernel[ky + 1][kx + 1];
                    weight += kernel[ky + 1][kx + 1];
                }
            }
        }
        d_output[y * width + x] = sum / weight;
    }
}