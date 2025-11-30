
// CUDA kernels 
extern "C" __global__ void compute_gradients_kernel(const float* I1, const float* I2, float*  Ix,  float* Iy, float* It, int w, int h, int stride);
extern "C" __global__ void hs_iteration_kernel(const float* Ix, const float* Iy, const float* It,
									const float* u, const float* v, float* uNew, float* vNew,
									int w, int h, int stride, float alpha2);
