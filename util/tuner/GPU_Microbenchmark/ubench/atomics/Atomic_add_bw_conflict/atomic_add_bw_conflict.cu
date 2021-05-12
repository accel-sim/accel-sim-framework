#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define REPEAT_TIMES 1024

template <class T>
__global__ void atomic_bw(uint32_t *startClk, uint32_t *stopClk, T *data1,
                          T *res) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t sum;
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  for (int j = 0; j < REPEAT_TIMES; ++j) {
    sum = sum + atomicAdd(&data1[0], 10);
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
  res[gid] = sum;
}

int main() {
  intilizeDeviceProp(0);

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  int32_t *data1 = (int32_t *)malloc(TOTAL_THREADS * sizeof(int32_t));
  int32_t *res = (int32_t *)malloc(TOTAL_THREADS * sizeof(int32_t));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  int32_t *data1_g;
  int32_t *res_g;

  for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
    data1[i] = (int32_t)i;
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&data1_g, TOTAL_THREADS * sizeof(int32_t)));
  gpuErrchk(cudaMalloc(&res_g, TOTAL_THREADS * sizeof(int32_t)));

  gpuErrchk(cudaMemcpy(data1_g, data1, TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyHostToDevice));

  atomic_bw<int32_t><<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(startClk_g, stopClk_g,
                                                        data1_g, res_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(res, res_g, TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyDeviceToHost));

  float bw;
  uint32_t total_time =
      *std::max_element(&stopClk[0], &stopClk[TOTAL_THREADS]) -
      *std::min_element(&startClk[0], &startClk[TOTAL_THREADS]);
  // uint32_t total_time = stopClk[0] - startClk[0];
  bw = ((float)(REPEAT_TIMES * TOTAL_THREADS * 4) / (float)(total_time));
  printf("Atomic int32 bandwidth = %f (byte/clk)\n", bw);
  printf("Total Clk number = %u \n", total_time);

  return 1;
}
