// assignment.cu
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>  // rand()
#include <ctime>    // time()

#define N 1000000   // total "pixels" in fake image (1M)

// ================== CUDA Error Checking ==================
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// ================== GPU Kernels ==================

// Non-branching: invert image
__global__ void invertKernel(const int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 255 - input[idx]; // always same operation
    }
}

// Branching: threshold filter
__global__ void thresholdKernel(const int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (input[idx] > 128)          // branching here
            output[idx] = 255;
        else
            output[idx] = 0;
    }
}

// ================== CPU Functions ==================
void invertHost(const int *input, int *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = 255 - input[i];
    }
}

void thresholdHost(const int *input, int *output, int n) {
    for (int i = 0; i < n; i++) {
        if (input[i] > 128)
            output[i] = 255;
        else
            output[i] = 0;
    }
}

// ================== Main ==================
int main(int argc, char* argv[]) {
    // Default config
    int blocks = 512;
    int threads = 256;

    // Parse command-line args
    if (argc == 3) {
        blocks = atoi(argv[1]);
        threads = atoi(argv[2]);
    }
    std::cout << "Running with " << blocks << " blocks, "
              << threads << " threads per block\n";

    // Allocate host memory
    int *h_in  = new int[N];
    int *h_out = new int[N];   // GPU results
    int *h_ref = new int[N];   // CPU results

    // Initialize with random pixel values [0,255]
    srand(time(0));
    for (int i = 0; i < N; i++) {
        h_in[i] = rand() % 256;
    }

    // Allocate device memory
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // ================== Non-Branching Test ==================
    std::cout << "\n=== Non-Branching: Invert Image ===\n";

    auto startGPU = std::chrono::high_resolution_clock::now();
    invertKernel<<<blocks, threads>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto stopGPU = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    auto startCPU = std::chrono::high_resolution_clock::now();
    invertHost(h_in, h_ref, N);
    auto stopCPU = std::chrono::high_resolution_clock::now();

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_ref[i]) {
            correct = false;
            break;
        }
    }

    auto gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(stopGPU - startGPU).count();
    auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(stopCPU - startCPU).count();

    std::cout << "GPU time: " << gpuTime << " us\n";
    std::cout << "CPU time: " << cpuTime << " us\n";
    std::cout << "Results match? " << (correct ? "YES" : "NO") << "\n";

    // ================== Branching Test ==================
    std::cout << "\n=== Branching: Threshold Filter ===\n";

    startGPU = std::chrono::high_resolution_clock::now();
    thresholdKernel<<<blocks, threads>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    stopGPU = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    startCPU = std::chrono::high_resolution_clock::now();
    thresholdHost(h_in, h_ref, N);
    stopCPU = std::chrono::high_resolution_clock::now();

    correct = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_ref[i]) {
            correct = false;
            break;
        }
    }

    gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(stopGPU - startGPU).count();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(stopCPU - startCPU).count();

    std::cout << "GPU time: " << gpuTime << " us\n";
    std::cout << "CPU time: " << cpuTime << " us\n";
    std::cout << "Results match? " << (correct ? "YES" : "NO") << "\n";

    // Cleanup
    delete[] h_in;
    delete[] h_out;
    delete[] h_ref;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

// build and run commands:
// nvcc -o assignment assignment.cu
// ./assignment
