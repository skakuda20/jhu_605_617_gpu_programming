# JHU 605 617 GPU Programming - Module 3 Assignment

## Description
This assignment contains a script that compares runtime of the same functions on the CPU vs the GPU.
There are two different functions, a non-branching function and a function that contains branching. 
The non-branching function "inverts" the values by subtracting the element from the randomly generated
data from the maximum value of 255. The branching function performs a thresholding function by sorting
the data by half of the maximum value, 127. The script prints the runtime of the functions on both the
CPU and GPU for each test.

## Usage
The code can be built by running `make`

and can be run by calling `./assignment <NUM_THREADS> <NUM_THREADS_PER_BLOCK>` where the two arguments are
option

## Analysis of CPU vs GPU Code
In the non-branching case, the GPU outperformed the CPU by about 9x, showing the advantage of parallel execution when all threads follow the same path. In the branching case, the GPU was over 100× faster than the CPU, even though conditional branching typically reduces GPU efficiency due to warp divergence. In this case, the simple branching operation did not introduce significant divergence, so the GPU’s parallelism still dominated. Overall, the GPU benefits from large-scale parallel workloads, while the CPU performance degrades more with branching.


## Analysis of Example Code

The example assignment shows a clear CPU vs GPU comparision using the chrono library to accurately measure
the runtime on each hardware type. The example code correctly handles memory management with the proper malloc, 
copy, and free calls. While the example main function tests straight vector addition, which is
a non-branching function, it fails to test and compare the performance of a similar function that does 
contain branching. The GPU timing is wrapped around add<<<...>>> without cudaDeviceSynchronize() which may
result in the timing being incorrect. 