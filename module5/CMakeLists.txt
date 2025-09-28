cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_erosion LANGUAGES CXX CUDA)

# Project Configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Source Files
set(SOURCES
    src/main.cu
    src/kernels.cu
)

set(HEADERS
    include/kernels.cuh
    src/utils.h
)

# Executable
add_executable(cuda_erosion ${SOURCES} ${HEADERS})

# Include Paths
target_include_directories(cuda_erosion PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# CUDA Options
set_target_properties(cuda_erosion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)