#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define REPEAT_TIMES 1024

template <class T>
__global__ void atmoic_latency(uint32_t *startClk, uint32_t *stopClk, T *data1,
                               T *res) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  // register T s1 = data1[gid];
  // register T s2 = data2[gid];
  // register T result = 0;
  uint32_t index = 0;
  int32_t offset = 10;
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");
  for (int j = 0; j < REPEAT_TIMES; ++j) {
    index = atomicAdd(&data1[index], offset);
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
  res[gid] = data1[0];
}

int main() {
  intilizeDeviceProp(0);

  THREADS_PER_BLOCK = 1;
  THREADS_PER_SM = 1;
  BLOCKS_NUM = 1;
  TOTAL_THREADS = 1;

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  int32_t *data1 = (int32_t *)malloc(REPEAT_TIMES * sizeof(int32_t));
  int32_t *res = (int32_t *)malloc(TOTAL_THREADS * sizeof(int32_t));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  int32_t *data1_g;
  int32_t *res_g;

  int32_t stride = 1;

  for (int32_t i = 0; i < (REPEAT_TIMES); i++)
    data1[i] = (i + stride) % REPEAT_TIMES;

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&data1_g, REPEAT_TIMES * sizeof(int32_t)));
  gpuErrchk(cudaMalloc(&res_g, TOTAL_THREADS * sizeof(int32_t)));
  gpuErrchk(cudaMemcpy(data1_g, data1, REPEAT_TIMES * sizeof(int32_t),
                       cudaMemcpyHostToDevice));

  atmoic_latency<int32_t><<<1, 1>>>(startClk_g, stopClk_g, data1_g, res_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(res, res_g, TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyDeviceToHost));

  float latency;
  latency = ((float)(stopClk[0] - startClk[0])) / ((float)(REPEAT_TIMES));
  printf("Atomic int32 latency = %f (clk)\n", latency);
  printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

  return 1;
}
