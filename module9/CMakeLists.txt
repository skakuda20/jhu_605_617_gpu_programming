cmake_minimum_required(VERSION 3.10)
project(noise_fft LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Add source files
set(SOURCES
    src/main.cu
    src/kernels.cu
)

# Find CUDA libraries
find_package(CUDA REQUIRED)

add_executable(noise_fft ${SOURCES})

target_link_libraries(noise_fft
    cufft
    curand
)

# Optionally, set extra NVCC flags
set_target_properties(noise_fft PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

target_compile_options(noise_fft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)