#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define REPEAT_TIMES 2048

template <class T>
__global__ void atomic_bw(uint64_t *startClk, uint64_t *stopClk, T *data1,
                          T *res) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  // register T s1 = data1[gid];
  // register T s2 = data2[gid];
  // register T result = 0;
  // synchronize all threads
  // int32_t res0, res1, res2, res3, res4, res5, res6, res7, res8, res9, res10,
  // res11, res12, res13, res14, res15;
  int32_t sum;
  asm volatile("bar.sync 0;");

  // start timing
  uint64_t start = clock64();

  for (uint32_t i = 0; i < REPEAT_TIMES; i++) {
    sum = sum + atomicAdd(&data1[(i * warpSize) + gid], 10);
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint64_t stop = clock64();

  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
  res[gid] = sum;
}

int main() {

  intilizeDeviceProp(0);
  unsigned ARRAY_SIZE = TOTAL_THREADS + (REPEAT_TIMES * WARP_SIZE);

  uint64_t *startClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));

  int32_t *res = (int32_t *)malloc(TOTAL_THREADS * sizeof(int32_t));
  int32_t *data1 = (int32_t *)malloc(ARRAY_SIZE * sizeof(int32_t));

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  int32_t *data1_g;
  int32_t *res_g;

  for (uint32_t i = 0; i < ARRAY_SIZE; i++) {
    data1[i] = (int32_t)i;
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&data1_g, ARRAY_SIZE * sizeof(int32_t)));
  gpuErrchk(cudaMalloc(&res_g, TOTAL_THREADS * sizeof(int32_t)));

  gpuErrchk(cudaMemcpy(data1_g, data1, ARRAY_SIZE * sizeof(int32_t),
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
  uint64_t total_time =
      *std::max_element(&stopClk[0], &stopClk[TOTAL_THREADS]) -
      *std::min_element(&startClk[0], &startClk[TOTAL_THREADS]);
  // uint64_t total_time = stopClk[0]-startClk[0];

  bw = (((float)REPEAT_TIMES * (float)TOTAL_THREADS * 4 * 8) /
        (float)(total_time));
  printf("Atomic int32 bandwidth = %f (byte/clk)\n", bw);
  printf("Total Clk number = %ld \n", total_time);

  return 1;
}
