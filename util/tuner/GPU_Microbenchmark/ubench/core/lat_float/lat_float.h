#ifndef LAT_FLOAT_DEF_H
#define LAT_FLOAT_DEF_H

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define REPEAT_TIMES 4096

template <class T>
__global__ void fpu_latency(uint32_t *startClk, uint32_t *stopClk, T *data1,
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
                 "fma.rn.f32 %0, %1, %2 , %0;\n\t"
                 "fma.rn.f32 %0, %1, %2 , %0;\n\t"
                 "fma.rn.f32 %0, %1, %2 , %0;\n\t"
                 "fma.rn.f32 %0, %1, %2 , %0;\n\t"
                 "}"
                 : "+f"(result), "+f"(s1), "+f"(s2));
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

float fpu_latency() {
  intilizeDeviceProp(0);

  THREADS_PER_BLOCK = 1;
  THREADS_PER_SM = 1;
  BLOCKS_NUM = 1;
  TOTAL_THREADS = 1;

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  float *data1 = (float *)malloc(TOTAL_THREADS * sizeof(float));
  float *data2 = (float *)malloc(TOTAL_THREADS * sizeof(float));
  float *res = (float *)malloc(TOTAL_THREADS * sizeof(float));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  float *data1_g;
  float *data2_g;
  float *res_g;

  for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
    data1[i] = (float)i;
    data2[i] = (float)i;
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&data1_g, TOTAL_THREADS * sizeof(float)));
  gpuErrchk(cudaMalloc(&data2_g, TOTAL_THREADS * sizeof(float)));
  gpuErrchk(cudaMalloc(&res_g, TOTAL_THREADS * sizeof(float)));

  gpuErrchk(cudaMemcpy(data1_g, data1, TOTAL_THREADS * sizeof(float),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(data2_g, data2, TOTAL_THREADS * sizeof(float),
                       cudaMemcpyHostToDevice));

  fpu_latency<float><<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
      startClk_g, stopClk_g, data1_g, data2_g, res_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(res, res_g, TOTAL_THREADS * sizeof(float),
                       cudaMemcpyDeviceToHost));

  float latency;
  latency = ((float)(stopClk[0] - startClk[0])) / ((float)(REPEAT_TIMES * 4));
  printf("float-precision FPU latency = %f (clk)\n", latency);
  printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

  return latency;
}

#endif
