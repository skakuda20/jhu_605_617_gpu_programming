# Module 12: OpenCL Image Convolution

This project demonstrates how to perform image convolution using OpenCL for GPU acceleration and OpenCV for image I/O. The program applies a convolution filter (such as blur) to an input image and writes the result to an output image. It is intended as a practical example for GPU programming and heterogeneous computing.

## Features
- Loads an image using OpenCV
- Runs a convolution kernel on the GPU using OpenCL
- Supports different convolution masks (blur, sharppen, edge)
- Saves the output image using OpenCV

## Prerequisites
- C++17 compiler
- OpenCL development libraries
- OpenCV (built with image I/O support)
- CMake

## Building

1. Make sure you have OpenCL and OpenCV installed on your system.
2. From the `module12` directory, run:

```bash
./scripts/build.sh
```

This will configure and build the project using CMake. The executable will be placed in `build/convolution`.

## Running

Use the provided run script, or run manually:

```bash
./build/convolution data/input.png data/output.png blur
```
- The first argument is the input image path.
- The second argument is the output image path.
- The third argument is the filter type (`blur`, `sharppen`, `edge`).

Example:
```bash
./build/convolution data/input.png data/output.png blur
```

## Files
- `src/main.cpp` — Main program logic
- `src/convolution.cl` — OpenCL kernel for convolution
- `scripts/build.sh` — Build script using CMake
- `scripts/run.sh` — Example run script

