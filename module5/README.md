# Module 5: GPU Programming

## Description
The goal of this assignment is to showcase the use of all forms of CUDA memory, including host memory,
global memory, shared memory, constant memory, and register memory. To demonstrate the different uses of memory, this assignment contains a script that applies erosion, a morphological operation commonly used in image 
processing. Erosion is used to remove pixels from the boundary of the input image thus shrinking objects
in the image. It uses a 2D matrix with an arbitrary size that will determine the effect of the erosion operation. Given a binary image, a pixel is set to 1 only if all of the neighboring pixels are also 1. This effectively
"shrinks" all of the objects in the image, also resulting in more defined edges.  


## Usage
This assingment can be built and run using the build script: `bash build.sh`

and can be run from the `build` directory by calling `./cuda_erosion <blockDimX> <blockDimY>` where the two arguments are optional. If no arguments are given, the default block size is 16x16.


### Output
- The program prints sample input and output values, and demonstrates CUDA memory usage.
- The block size can be changed to experiment with performance and correctness.
