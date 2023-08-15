// This benchmark measures the maximum read bandwidth of shared memory and L1 at
// the same time

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

// array size is half the L1 size (2) * float size (4)
#define ARRAY_SIZE (L1_SIZE / 8)
// 32 KB of shd memory
#define SHARED_MEM_SIZE (32 * 1024 / 4)
#define ITERS 4096

__global__ void shared_bw(uint32_t *startClk, uint32_t *stopClk,
                          uint32_t *dsink, uint32_t *l1, uint32_t stride) {

  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;
  uint32_t n_threads = blockDim.x * gridDim.x;

  register uint32_t tmp_s = uid;
  register uint32_t tmp_l1 = uid;
  uint32_t start = 0;
  uint32_t stop = 0;

  __shared__ uint32_t s[SHARED_MEM_SIZE]; // static shared memory
  // uint32_t s[SHARED_MEM_SIZE];
  // one thread to initialize the pointer-chasing array
  for (uint32_t i = uid; i < (SHARED_MEM_SIZE); i += n_threads)
    s[i] = (i + stride + 7) % SHARED_MEM_SIZE;

  // warmup l1 cache
  for (uint32_t i = 0; i < ARRAY_SIZE; ++i) {
    tmp_l1 = l1[tmp_l1];
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  // load data from l1 cache and accumulate
  for (uint32_t i = 0; i < ITERS; ++i) {
    tmp_s = s[tmp_s];
    tmp_l1 = l1[tmp_l1];
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  // sink0 = tmp;
  // write time and data back to memory
  startClk[uid] = start;
  stopClk[uid] = stop;
  dsink[uid] = tmp_s + tmp_l1;
}

int main() {
  intilizeDeviceProp(0);

  BLOCKS_NUM = 1;
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;
  THREADS_PER_SM = THREADS_PER_BLOCK * BLOCKS_NUM;

  assert(SHARED_MEM_SIZE * sizeof(uint32_t) < MAX_SHARED_MEM_SIZE_PER_BLOCK);

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *dsink = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));

  uint32_t *posArray = (uint32_t *)malloc(ARRAY_SIZE * sizeof(uint32_t));

  uint32_t stride = 1024;

  for (uint32_t i = 0; i < ARRAY_SIZE; i++)
    posArray[i] = (i + stride + 1) % ARRAY_SIZE;

  uint32_t *posArray_g;

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  uint32_t *dsink_g;

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&dsink_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&posArray_g, ARRAY_SIZE * sizeof(uint32_t)));

  gpuErrchk(cudaMemcpy(posArray_g, posArray, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));

  shared_bw<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g,
                                               posArray_g, stride);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(dsink, dsink_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));

  double bw;
  bw = (double)(ITERS * TOTAL_THREADS * 4 * 2) /
       ((double)(stopClk[0] - startClk[0]));
  printf("Shared Memory Bandwidth = %f (byte/clk/SM)\n", bw);
  printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

  return 1;
}
