// main.cu
#include "kernels.cuh"
#include "utils.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

int main() {
    // Generate random image data
    int width = 512;
    int height = 512;
    int imgSize = width * height;
    int* image = new int[imgSize];
    srand(time(0));
    for (int i = 0; i < imgSize; ++i) {
        image[i] = rand() % 256; // Simulate grayscale image
    }

    // Allocate host memory for output
    int* host_output = new int[imgSize];

    // Allocate device global memory 
    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, imgSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, imgSize * sizeof(int)));

    // Copy data host to device
    CUDA_CHECK(cudaMemcpy(d_input, image, imgSize * sizeof(int), cudaMemcpyHostToDevice));

    // Copy structuring element into constant memory
    int kernelMask[9] = {1,1,1, 1,1,1, 1,1,1};
    copy_to_constant_memory(kernelMask);

    // Configure kernel launch
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    // Launch CUDA erosion kernel
    erosionKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result device to host
    CUDA_CHECK(cudaMemcpy(host_output, d_output, imgSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Save result (optional, here just print a few values)
    std::cout << "Sample output values: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << host_output[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] image;
    delete[] host_output;

    return 0;
}
