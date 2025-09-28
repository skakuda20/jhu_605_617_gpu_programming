// main.cu
#include "kernels.cuh"
#include "utils.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

int main(int argc, char* argv[]) {
    // Host memory 
    std::cout << "[Host] image and host_output are allocated on the host (CPU)\n";
    
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

    // Allocate host memory for output
    int* host_output = new int[imgSize];

    // Allocate device global memory 
    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, imgSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, imgSize * sizeof(int)));
    std::cout << "[Global] d_input and d_output are allocated in device global memory (GPU)\n";

    // Copy data host to device
    CUDA_CHECK(cudaMemcpy(d_input, image, imgSize * sizeof(int), cudaMemcpyHostToDevice));
    std::cout << "[Global] Copied image from host to device global memory (d_input)\n";

    // Copy structuring element into constant memory
    int kernelMask[9] = {1,1,1, 1,1,1, 1,1,1};

    // Print structuring element
    std::cout << "Structuring element: ";
    for (int i = 0; i < 9; ++i) {
        std::cout << kernelMask[i] << " ";
    }
    std::cout << std::endl;
    copy_to_constant_memory(kernelMask);
    std::cout << "[Constant] kernelMask copied to device constant memory (d_structuringElement)\n";

    // Configure kernel launch
    dim3 blockSize(blockDimX, blockDimY);
    dim3 gridSize((width + blockDimX - 1) / blockDimX, (height + blockDimY - 1) / blockDimY);
    std::cout << "Block size: (" << blockDimX << ", " << blockDimY << ")\n";
    std::cout << "Grid size: (" << gridSize.x << ", " << gridSize.y << ")\n";

    // Calculate shared memory size
    int radius = 1;
    int sharedMemSize = (blockDimX + 2 * radius) * (blockDimY + 2 * radius) * sizeof(int);

    // Launch CUDA erosion kernel with dynamic shared memory
    erosionKernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "[Shared/Register] See kernel output below for shared/register memory demonstration.\n";

    // Copy result device to host
    CUDA_CHECK(cudaMemcpy(host_output, d_output, imgSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Save result
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
