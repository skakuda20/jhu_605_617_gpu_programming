#!/bin/bash
set -e

# Create build directory
mkdir -p build
cd build

# Run CMake and build
cmake ..
make -j$(nproc)

# Run the executable
./noise_fft