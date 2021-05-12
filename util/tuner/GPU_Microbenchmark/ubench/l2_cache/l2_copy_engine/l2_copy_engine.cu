// This ubench meaures if DMA memory copy is cached in L2 by default

#include <assert.h>
#include <cuda.h>
#include <iostream> // std::abs
#include <math.h>   // std::abs
#include <numeric>  // std::accumulate
#include <stdio.h>
#include <stdlib.h>
using namespace std;

#include "../../../hw_def/hw_def.h"
#include "../l2_lat/l2_lat.h"

#define REPEAT_TIMES 32768 // iterate over the array ITERS times
#define ARRAY_SIZE_L2 32768

__global__ void l2_lat_no_warmpu(uint32_t *startClk, uint32_t *stopClk,
                                 uint64_t *posArray, uint64_t *dsink) {

  // thread index
  uint32_t tid = threadIdx.x;

  // do pointer-chasing without warmpup
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
    for (uint32_t i = 0; i < REPEAT_TIMES; ++i) {
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

int main() {
  intilizeDeviceProp(0);

  unsigned THREADS_NUM = 1;

  // Array size must not exceed L2 size
  assert(ARRAY_SIZE_L2 * sizeof(uint64_t) < L2_SIZE);

  uint64_t *posArray = (uint64_t *)malloc(ARRAY_SIZE_L2 * sizeof(uint64_t));
  uint32_t *startClk = (uint32_t *)malloc(THREADS_NUM * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(THREADS_NUM * sizeof(uint32_t));
  uint64_t *dsink = (uint64_t *)malloc(THREADS_NUM * sizeof(uint64_t));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  uint64_t *posArray_g;
  uint64_t *dsink_g;

  uint64_t stride = 1;

  gpuErrchk(cudaMalloc(&startClk_g, THREADS_NUM * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, THREADS_NUM * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&posArray_g, ARRAY_SIZE_L2 * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&dsink_g, THREADS_NUM * sizeof(uint64_t)));

  // initilze pointer-chasing on the CPU side
  for (uint64_t i = 0; i < ARRAY_SIZE_L2; i++) {
    uint64_t *tmp = posArray_g + ((i + stride) % ARRAY_SIZE_L2);
    posArray[i] = (uint64_t)tmp;
  }

  gpuErrchk(cudaMemcpy(posArray_g, posArray, sizeof(uint64_t) * ARRAY_SIZE_L2,
                       cudaMemcpyHostToDevice));

  // here we measure the latency of the request without warmup
  l2_lat_no_warmpu<<<1, THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g,
                                       dsink_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, THREADS_NUM * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, THREADS_NUM * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(dsink, dsink_g, THREADS_NUM * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  float l2_nowarmp_lat = (float)(stopClk[0] - startClk[0]) / REPEAT_TIMES;
  printf("L2 Latency no-warmp up = %12.4f cycles \n", l2_nowarmp_lat);
  printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

  // then we measure L2 hit latncy with warmpup
  float l2_hit_lat2 = l2_hit_lat();

  // if the latency is close to the l2 hit latency, then the memcpy are cached
  // by default at L2
  float error = (abs(l2_nowarmp_lat - l2_hit_lat2) / l2_hit_lat2) * 100;
  bool cached = (error < 10.0f);
  if (cached)
    printf("Is memcpy cached in L2? Yes, error=%2.1f\n", error);
  else
    printf("Is memcpy cached in L2? No, error=%2.1f\n", error);

  if (ACCEL_SIM_MODE) {
    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-gpgpu_perf_sim_memcpy " << cached << std::endl;
  }

  return 1;
}
