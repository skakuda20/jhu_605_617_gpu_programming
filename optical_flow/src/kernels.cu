#include "kernels.cuh"

__global__ void compute_gradients_kernel(const float* I1, const float* I2, float*  Ix,  float* Iy, float* It, int w, int h, int stride) {
  // Compute image gradients Ix, Iy, It
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x <= 0 ||  y <= 0 || x >= w-1 || y >= h-1) return;

  // central differences
  int idx = y*stride + x;
  float i1m = I1[y*stride + (x-1)];
  float i1p = I1[y*stride + (x+1)];
  float i2m = I2[y*stride + (x-1)];
  float i2p = I2[y*stride + (x+1)];
  float ix = (i1p - i1m + i2p - i2m) * 0.25f;
  float i1um = I1[(y-1)*stride + x];
  float i1dm = I1[(y+1)*stride + x];
  float i2um = I2[(y-1)*stride + x];
  float i2dm = I2[(y+1)*stride + x];
  float iy = (i1dm - i1um + i2dm - i2um) * 0.25f;
  float it = I2[idx] - I1[idx];
  Ix[idx] = ix; Iy[idx] = iy; It[idx] = it;


}

// One iteration update kernel
__global__ void hs_iteration_kernel(const float* Ix, const float* Iy, const float* It,
                                    const float* u, const float* v, float* uNew, float* vNew,
                                    int w, int h, int stride, float alpha2) {
    // Compute local averages
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= w-1 || y >= h-1) return;
    int idx = y*stride + x;

    // local average of u,v
    float uBar = (u[idx-1] + u[idx+1] + u[idx-stride] + u[idx+stride]) * 0.25f;
    float vBar = (v[idx-1] + v[idx+1] + v[idx-stride] + v[idx+stride]) * 0.25f;
    float ix = Ix[idx], iy = Iy[idx], it = It[idx];
    float denom = alpha2 + ix*ix + iy*iy;
    float term = (ix * uBar + iy * vBar + it);
    float du = - (ix * term) / denom;
    float dv = - (iy * term) / denom;
    uNew[idx] = uBar + du;
    vNew[idx] = vBar + dv;
}
