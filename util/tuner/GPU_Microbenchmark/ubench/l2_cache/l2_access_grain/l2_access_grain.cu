/*
This benchmark measures l2 access granularity for differnet strides
check the nvprof or nvsight for received l2 reads and writes
for further details, see our arvix paper: https://arxiv.org/pdf/1810.07269.pdf

Compile this file using the following command to disable L1 cache:
    nvcc -Xptxas -dlcm=cg l2_sector_grain.cu

run the program with nsight
 make nvsight ./l2_access_grain

 Result: All Nvidia HW generation since kepler has 32B access granularity
 */

#include <assert.h>
#include <cstdio>
#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device code
__global__ void l2_stride_cons(const float *A, float *C, int stride)

{

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  C[i * stride] = A[i * stride];
}

__global__ void l2_stride(const float *A, float *C, int stride)

{

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  C[((i / stride) * 32) + (i % stride)] = A[((i / stride) * 32) + (i % stride)];
}

// Host code
void l2_stride(int N, int threadsPerBlock, int stride) {
  // Variables
  float *h_A;
  float *h_C;

  float *d_A;
  float *d_C;

  size_t size = N * sizeof(float) * 32;

  // Allocate input vectors h_A and h_B in host memory
  h_A = (float *)malloc(size);
  h_C = (float *)malloc(size);

  // fill array
  for (uint32_t i = 0; i < N; i++)
    h_A[i] = (float)i;

  // Allocate vectors in device memory
  gpuErrchk(cudaMalloc((void **)&d_A, size));
  gpuErrchk(cudaMalloc((void **)&d_C, size));

  // Copy vectors from host memory to device memory
  gpuErrchk(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

  // Invoke kernel
  int blocksPerGrid = ((N + threadsPerBlock - 1) / threadsPerBlock);

  l2_stride<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, stride);
  gpuErrchk(cudaPeekAtLastError());

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  gpuErrchk(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  // Free device memory
  if (d_A)
    cudaFree(d_A);
  if (d_C)
    cudaFree(d_C);

  // Free host memory
  if (h_A)
    free(h_A);
  if (h_C)
    free(h_C);
}
//////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  intilizeDeviceProp(0);

  for (int i = 1; i <= WARP_SIZE; ++i)
    l2_stride(WARP_SIZE, WARP_SIZE, i);

  std::cout << "\nThis benchmark measures l2 access granularity for differnet "
               "strides.\n";
  std::cout << "check the nvprof or nvsight for received l2 reads and write.\n";
  std::cout
      << "to run the program with nsight: make nvsight ./l2_access_grain\n";
  std::cout << "stats to look at: lts__t_sectors_srcunit_tex_op_read.sum and "
               "lts__t_sectors_srcunit_tex_op_write.sum \n\n";
}
