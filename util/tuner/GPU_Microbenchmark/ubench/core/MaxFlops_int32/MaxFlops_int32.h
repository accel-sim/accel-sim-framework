#ifndef MAXFLOPS_INT32_DEF_H
#define MAXFLOPS_INT32_DEF_H

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define REPEAT_TIMES 1024

template <class T>
__global__ void max_flops(uint32_t *startClk, uint32_t *stopClk, T *data1,
                          T *data2, T *res) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s1 = data1[gid];
  register T s2 = data2[gid];
  register T result = 0;

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  for (int j = 0; j < REPEAT_TIMES; ++j) {
    asm volatile("{\t\n"
                 "mad.lo.s32 %0, %1, %2 , %0;\n\t"
                 "mad.lo.s32 %0, %1, %2 , %0;\n\t"
                 "mad.lo.s32 %0, %1, %2 , %0;\n\t"
                 "mad.lo.s32 %0, %1, %2 , %0;\n\t"
                 "}"
                 : "+r"(result), "+r"(s1), "+r"(s2));
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
  res[gid] = result;
}

float max_int32_flops() {
  intilizeDeviceProp(0);

  BLOCKS_NUM = 1;
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  int32_t *data1 = (int32_t *)malloc(TOTAL_THREADS * sizeof(int32_t));
  int32_t *data2 = (int32_t *)malloc(TOTAL_THREADS * sizeof(int32_t));
  int32_t *res = (int32_t *)malloc(TOTAL_THREADS * sizeof(int32_t));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  int32_t *data1_g;
  int32_t *data2_g;
  int32_t *res_g;

  for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
    data1[i] = (int32_t)i;
    data2[i] = (int32_t)i;
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&data1_g, TOTAL_THREADS * sizeof(int32_t)));
  gpuErrchk(cudaMalloc(&data2_g, TOTAL_THREADS * sizeof(int32_t)));
  gpuErrchk(cudaMalloc(&res_g, TOTAL_THREADS * sizeof(int32_t)));

  gpuErrchk(cudaMemcpy(data1_g, data1, TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(data2_g, data2, TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyHostToDevice));

  max_flops<int32_t><<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
      startClk_g, stopClk_g, data1_g, data2_g, res_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(res, res_g, TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyDeviceToHost));

  float flops;
  flops = (float)(REPEAT_TIMES * TOTAL_THREADS * 8) /
          ((float)(stopClk[0] - startClk[0]));
  printf("int32 FLOP per SM = %f (flop/clk/SM)\n", flops);
  printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

  return flops;
}

#endif