// filepath: /home/kakudas/GitHub/EN605.617/module9/src/kernels.cuh
#pragma once
#include <cufft.h>
__global__ void applyGaussianFilter(cufftComplex* freqData, int width, int height, float sigma);