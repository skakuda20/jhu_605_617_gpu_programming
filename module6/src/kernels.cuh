// // kernels.cuh
__global__ void gaussianBlurKernel(int* d_input, int* d_output, int width, int height);
__global__ void blurKernel(int* d_input, int* d_output, int width, int height);
__global__ void invertKernel(int* d_input, int* d_output, int width, int height);
__global__ void erosionKernel(int* d_input, int* d_output, int width, int height, int radius);

extern __constant__ int d_structuringElement[9];

