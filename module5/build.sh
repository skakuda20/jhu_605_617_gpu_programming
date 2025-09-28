#!/bin/bash
set -e

# Go to script directory
cd "$(dirname "$0")"

# Create build directory if it does not exist
if [ ! -d build ]; then
    mkdir build
fi
cd build

# Configure and build
cmake ..
make

# Run with different block sizes
for bx in 8 16 32; do
  for by in 8 16 32; do
  echo -e "\nRunning with block size: $bx x $by"
    ./cuda_erosion $bx $by
  done
done
