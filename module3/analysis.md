## Analysis of CPU vs GPU Code
In the non-branching case, the GPU outperformed the CPU by about 9x, showing the advantage of parallel execution when all threads follow the same path. In the branching case, the GPU was over 100× faster than the CPU, even though conditional branching typically reduces GPU efficiency due to warp divergence. In this case, the simple branching operation did not introduce significant divergence, so the GPU’s parallelism still dominated. Overall, the GPU benefits from large-scale parallel workloads, while the CPU performance degrades more with branching.


## Analysis of Example Code

The example assignment shows a clear CPU vs GPU comparision using the chrono library to accurately measure
the runtime on each hardware type. The example code correctly handles memory management with the proper malloc, 
copy, and free calls. While the example main function tests straight vector addition, which is
a non-branching function, it fails to test and compare the performance of a similar function that does 
contain branching. The GPU timing is wrapped around add<<<...>>> without cudaDeviceSynchronize() which may
result in the timing being incorrect. 