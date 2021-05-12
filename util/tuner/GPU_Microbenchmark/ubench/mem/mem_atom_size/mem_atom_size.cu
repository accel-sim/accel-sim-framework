/*
This benchmark measures mem atom size
check the nvprof or nvsight for received mem reads and writes
for further details, see our arvix paper: https://arxiv.org/pdf/1810.07269.pdf

Compile this file using the following command to disable L1 cache:
    nvcc -Xptxas -dlcm=cg mem_atom_size.cu

run the program with nsight
 make nvsight ./mem_atom_size

Result: many Nvidia HW generation since kepler has 32B mem_atom_size granularity
However, It seems some pascal and volta GPUs have atom size of 64B larger than
the L2 access grain (32B). We asked Nvidia about this weird behavior, and that's
their reply: https://forums.developer.nvidia.com/t/pascal-l1-cache/49571/15
*/

#include <assert.h>
#include <cstdio>
#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device code
__global__ void mem_stride_cons(const float *A, float *C, int stride)

{

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  C[i * stride] = A[i * stride];
}

// Host code
void mem_stride(int N, int threadsPerBlock, int stride) {
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

  mem_stride_cons<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, stride);
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

  // make the array very large to avoid L2 cache resident
  // run the threads with stride of 32 (128B) to avoid any coalescing
  mem_stride((L2_SIZE / sizeof(float)) * 2, 256, 32);

  std::cout << std::endl
            << "This benchmark measures mem atom size granularity" << std::endl;

  std::cout << "check the nvprof or nvsight for received mem reads and writes"
            << std::endl;
  std::cout << "to run the program with nsight: make nvsight ./l2_access_grain"
            << std::endl;
  std::cout
      << "stats to look at: dram__sectors_read.sum & dram__sectors_write.sum & "
         "dram__bytes_read.sum & dram__sectors_read.sum"
      << std::endl
      << std::endl;

  std::cout
      << "we launched " << (L2_SIZE / sizeof(float)) * 2
      << " read memory reqs (1 req per thread) with a stride of 32 (128 bytes)"
      << std::endl;
  std::cout << "if the number of memory reads is the same as read reqs, then "
               "mem atom size is 32B"
            << std::endl;
  std::cout << "if the number of memory reads is 2X issued read reqs, then mem "
               "atom size is 64B, etc."
            << std::endl
            << std::endl;
}
