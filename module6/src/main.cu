// main.cu
#include "kernels.cuh"
#include "utils.h"
#include <cstdlib>
#include <ctime>
#include <iostream>


int main(int argc, char* argv[]) {

    // Parse command line arguments for block size
    int blockDimX = 16;
    int blockDimY = 16;
    if (argc == 3) {
        blockDimX = atoi(argv[1]);
        blockDimY = atoi(argv[2]);
    }

    // Generate random image data
    int width = 512;
    int height = 512;
    int imgSize = width * height;
    int* image = new int[imgSize];
    srand(time(0));
    for (int i = 0; i < imgSize; ++i) {
        image[i] = rand() % 256; // Simulate grayscale image
    }

    // Allocate device memory
    int *d_input, *d_blurOut, *d_errosionOut;
    size_t imageBytes = imgSize * sizeof(int);
    
    std::cout << "Block size: (" << blockDimX << ", " << blockDimY << ")\n";
    std::cout << "Grid size: (" << gridSize.x << ", " << gridSize.y << ")\n";

    std::cout << "Allocating device memory...\n";
    CUDA_CHECK(cudaMalloc(&d_input, imageBytes));
    CUDA_CHECK(cudaMalloc(&d_blurOut, imageBytes));
    CUDA_CHECK(cudaMalloc(&d_errosionOut, imageBytes));

    std::cout << "Copying input image to device...\n";
    CUDA_CHECK(cudaMemcpy(d_input, image, imageBytes, cudaMemcpyHostToDevice));

    std::cout << "Creating CUDA streams and events...\n";
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Launching gaussianBlurKernel and erosionKernel...\n";
    CUDA_CHECK(cudaEventRecord(start));
    gaussianBlurKernel<<<gridSize, blockSize, 0, stream1>>>(d_input, d_blurOut, width, height);
    erosionKernel<<<gridSize, blockSize, (blockDimX + 2) * (blockDimY + 2) * sizeof(int), stream2>>>(d_input, d_errosionOut, width, height, 1);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    std::cout << "Kernels finished executing.\n";

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Time elapsed for concurrent execution: " << milliseconds << " ms\n";

    std::cout << "Cleaning up resources...\n";
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_blurOut));
    CUDA_CHECK(cudaFree(d_errosionOut));
    delete[] image;
    std::cout << "Done.\n";
    return 0;
}