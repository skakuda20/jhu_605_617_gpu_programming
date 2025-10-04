cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_image_kernels LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    src/main.cu
    src/kernels.cu
)

set(HEADERS
    src/kernels.cuh
    src/utils.h
)

# Executable
add_executable(cuda_image_kernels ${SOURCES})

# Include directories
target_include_directories(cuda_image_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# CUDA options
set_target_properties(cuda_image_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75 # Adjust to your GPU
)

