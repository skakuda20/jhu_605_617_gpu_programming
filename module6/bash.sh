#!/bin/bash

set -e

# Build the project using CMake
BUILD_DIR="build"
EXECUTABLE="cuda_image_kernels"

# Clean previous build
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

cmake ..
make -j

cd ..

# Define block/thread sizes to test
BLOCK_SIZES=(8 16 32)
THREAD_SIZES=(8 16 32)

echo "Testing different block/thread sizes:"
for bx in "${BLOCK_SIZES[@]}"; do
    for by in "${THREAD_SIZES[@]}"; do
        echo "----------------------------------------"
        echo "Running with blockDimX=$bx blockDimY=$by"
        ./$BUILD_DIR/$EXECUTABLE $bx $by
        echo "----------------------------------------"
    done
done