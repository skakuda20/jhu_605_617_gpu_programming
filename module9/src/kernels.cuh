// kernels.cuh

__global__ void applyGaussianFilter(cufftComplex* freqData, int width, int height, float sigma);
