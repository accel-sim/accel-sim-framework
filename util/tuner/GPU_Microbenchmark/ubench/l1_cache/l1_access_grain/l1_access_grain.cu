/*
This benchmark measures L1 coalescing granularity for differnet strides
check the nvprof or nvsight for received l1 reads and writes
for further details, see our arvix paper: https://arxiv.org/pdf/1810.07269.pdf

run the program with nsight
 make nvsight ./l1_access_grain

*/

#include <cstdio>
#include <iostream>

using namespace std;

#include "../../../hw_def/hw_def.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device code
__global__ void l1_stride_cons(const float *A, float *C, int stride)

{

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  C[i * stride] = A[i * stride];
}

__global__ void l1_stride(const float *A, float *C, int stride)

{

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  C[((i / stride) * 32) + (i % stride)] = A[((i / stride) * 32) + (i % stride)];
}

// Host code
void coaslescer_stride(int N, int threadsPerBlock, int stride) {
  // Variables
  float *h_A;
  float *h_C;

  float *d_A;
  float *d_C;

  size_t size = N * sizeof(float) * 32;

  // Allocate input vectors h_A and h_B in host memory
  h_A = (float *)malloc(size);
  h_C = (float *)malloc(size);

  // Initialize input vectors
  for (uint32_t i = 0; i < N; i++)
    h_A[i] = (float)i;

  // Allocate vectors in device memory
  gpuErrchk(cudaMalloc((void **)&d_A, size));
  gpuErrchk(cudaMalloc((void **)&d_C, size));

  // Copy vectors from host memory to device memory
  gpuErrchk(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

  // Invoke kernel
  int blocksPerGrid = ((N + threadsPerBlock - 1) / threadsPerBlock);

  l1_stride<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, stride);
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

  for (int i = 1; i <= WARP_SIZE; ++i) {
    coaslescer_stride(WARP_SIZE, WARP_SIZE, i);
  }

  std::cout << "\nThis benchmark measures coalescing granularity for differnet "
               "strides.\n";
  std::cout
      << "check the nvprof or nvsight for received l1 reads and writes.\n";
  std::cout
      << "to run the program with nsight: make nvsight ./l1_access_grain\n";
  std::cout
      << "stats to look at: l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum & "
         "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum\n\n";
}
