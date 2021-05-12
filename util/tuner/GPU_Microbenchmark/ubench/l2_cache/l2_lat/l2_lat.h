// This code is a modification of L1 cache benchmark from
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking":
// https://arxiv.org/pdf/1804.06826.pdf

// This benchmark measures the latency of L2 latency using pointer-chasing

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include "../../../hw_def/hw_def.h"

#define ITERS 32768 // iterate over the array ITERS times
#define ARRAY_SIZE 4096

__global__ void l2_hit_lat(uint32_t *startClk, uint32_t *stopClk,
                           uint64_t *posArray, uint64_t *dsink) {

  // thread index
  uint32_t tid = threadIdx.x;

  // initialize pointer-chasing array with just one thread
  // warp up L2 cache and ensure all next accesses hit
  if (tid == 0) {
    for (uint32_t i = 0; i < (ARRAY_SIZE - 1); i++)
      posArray[i] = (uint64_t)(posArray + i + 1);

    posArray[ARRAY_SIZE - 1] = (uint64_t)posArray;
  }

  if (tid == 0) {

    uint64_t *ptr = posArray + tid;
    uint64_t ptr1, ptr0;

    // initialize the pointers with the start address
    // use cg modifier to cache the load in L2 and bypass L1
    asm volatile("{\t\n"
                 "ld.global.cg.u64 %0, [%1];\n\t"
                 "}"
                 : "=l"(ptr1)
                 : "l"(ptr)
                 : "memory");

    // synchronize all threads
    asm volatile("bar.sync 0;");

    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

    // pointer-chasing ITERS times
    // use cg modifier to cache the load in L2 and bypass L1
    for (uint32_t i = 0; i < ITERS; ++i) {
      asm volatile("{\t\n"
                   "ld.global.cg.u64 %0, [%1];\n\t"
                   "}"
                   : "=l"(ptr0)
                   : "l"((uint64_t *)ptr1)
                   : "memory");
      ptr1 = ptr0; // swap the register for the next load
    }

    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

    // write time and data back to memory
    startClk[tid] = start;
    stopClk[tid] = stop;
    dsink[tid] = ptr1;
  }
}

int l2_hit_lat() {
  intilizeDeviceProp(0);

  BLOCKS_NUM = 1;
  THREADS_PER_BLOCK = 1;
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  // Array size must not exceed L2 size
  assert(ARRAY_SIZE * sizeof(uint64_t) < L2_SIZE);

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint64_t *dsink = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  uint64_t *posArray_g;
  uint64_t *dsink_g;

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&posArray_g, ARRAY_SIZE * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&dsink_g, TOTAL_THREADS * sizeof(uint64_t)));

  l2_hit_lat<<<1, THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, posArray_g,
                                       dsink_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(dsink, dsink_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));

  float lat = (float)(stopClk[0] - startClk[0]) / ITERS;
  printf("L2 Hit Latency = %12.4f cycles \n", lat);
  printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

  return lat;
}
