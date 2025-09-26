// utils.h
// declare function load_image(filename) -> (image_array, width, height)
// declare function save_image(filename, image_array, width, height)

#pragma once
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}