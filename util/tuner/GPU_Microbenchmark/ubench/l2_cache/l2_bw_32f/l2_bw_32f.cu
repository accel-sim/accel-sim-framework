// This code is a modification of L2 cache benchmark from
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking":
// https://arxiv.org/pdf/1804.06826.pdf

// This benchmark measures the maximum read bandwidth of L2 cache for 32f
// Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bw.cu

#include <algorithm>
#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define REPEAT_TIMES 2048

/*
L2 cache is warmed up by loading posArray and adding sink
Start timing after warming up
Load posArray and add sink to generate read traffic
Repeat the previous step while offsetting posArray by one each iteration
Stop timing and store data
*/

__global__ void l2_bw(uint64_t *startClk, uint64_t *stopClk, float *dsink,
                      float *posArray, unsigned ARRAY_SIZE) {
  // block and thread index
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;

  // a register to avoid compiler optimization
  float sink = 0;

  // warm up l2 cache
  for (uint32_t i = uid; i < ARRAY_SIZE; i += blockDim.x * gridDim.x) {
    float *ptr = posArray + i;
    // every warp loads all data in l2 cache
    // use cg modifier to cache the load in L2 and bypass L1
    asm volatile("{\t\n"
                 ".reg .f32 data;\n\t"
                 "ld.global.cg.f32 data, [%1];\n\t"
                 "add.f32 %0, data, %0;\n\t"
                 "}"
                 : "+f"(sink)
                 : "l"(ptr)
                 : "memory");
  }

  asm volatile("bar.sync 0;");

  // start timing
  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  // load data from l2 cache and accumulate,
  for (uint32_t i = 0; i < REPEAT_TIMES; i++) {
    float *ptr = posArray + (i * warpSize) + uid;
    asm volatile("{\t\n"
                 ".reg .f32 data;\n\t"
                 "ld.global.cg.f32 data, [%1];\n\t"
                 "add.f32 %0, data, %0;\n\t"
                 "}"
                 : "+f"(sink)
                 : "l"(ptr)
                 : "memory");
  }
  asm volatile("bar.sync 0;");

  // stop timing
  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  // store the result
  startClk[bid * blockDim.x + tid] = start;
  stopClk[bid * blockDim.x + tid] = stop;
  dsink[bid * blockDim.x + tid] = sink;
}

int main() {
  intilizeDeviceProp(0);

  unsigned ARRAY_SIZE = TOTAL_THREADS + REPEAT_TIMES * WARP_SIZE;
  assert(ARRAY_SIZE * sizeof(float) <
         L2_SIZE); // Array size must not exceed L2 size

  uint64_t *startClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));

  float *posArray = (float *)malloc(ARRAY_SIZE * sizeof(float));
  float *dsink = (float *)malloc(TOTAL_THREADS * sizeof(float));

  float *posArray_g;
  float *dsink_g;
  uint64_t *startClk_g;
  uint64_t *stopClk_g;

  for (int i = 0; i < ARRAY_SIZE; i++)
    posArray[i] = (float)i;

  gpuErrchk(cudaMalloc(&posArray_g, ARRAY_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc(&dsink_g, TOTAL_THREADS * sizeof(float)));
  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));

  gpuErrchk(cudaMemcpy(posArray_g, posArray, ARRAY_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice));

  l2_bw<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g,
                                           posArray_g, ARRAY_SIZE);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(dsink, dsink_g, TOTAL_THREADS * sizeof(float),
                       cudaMemcpyDeviceToHost));

  float bw, BW;
  unsigned long long data =
      (unsigned long long)TOTAL_THREADS * REPEAT_TIMES * sizeof(float);
  uint64_t total_time = stopClk[0] - startClk[0];
  // uint64_t total_time =
  // *std::max_element(&stopClk[0],&stopClk[TOTAL_THREADS])-*std::min_element(&startClk[0],&startClk[TOTAL_THREADS]);
  bw = (float)(data) / ((float)(total_time));
  BW = bw * CLK_FREQUENCY * 1000000 / 1024 / 1024 / 1024;
  std::cout << "L2 bandwidth = " << bw << "(byte/clk), " << BW << "(GB/s)\n";
  float max_bw = get_num_channels(MEM_BITWIDTH, DRAM_MODEL) *
                 L2_BANKS_PER_MEM_CHANNEL * L2_BANK_WIDTH_in_BYTE;
  BW = max_bw * CLK_FREQUENCY * 1000000 / 1024 / 1024 / 1024;
  std::cout << "Max Theortical L2 bandwidth = " << max_bw << "(byte/clk), "
            << BW << "(GB/s)\n";
  std::cout << "L2 BW achievable = " << (bw / max_bw) * 100 << "%\n";
  std::cout << "Total Clk number = " << total_time << "\n";

  return 1;
}
