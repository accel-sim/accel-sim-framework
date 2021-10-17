#ifndef LAT_TENSOR_DEF_H
#define LAT_TENSOR_DEF_H

#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define REPEAT_ITERS 4096

#define M_SIZE 16 * 16

using namespace nvcuda;

template <class T, class R>
__global__ void tensor_latency(uint64_t *startClk, uint64_t *stopClk, T *a,
                               T *b, R *res) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // register T result = 0;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, R> c_frag;

  wmma::load_matrix_sync(a_frag, a, 16);
  wmma::fill_fragment(c_frag, 0.0f);
  wmma::load_matrix_sync(b_frag, b, 16);

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  for (int j = 0; j < REPEAT_ITERS; ++j) {
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  wmma::store_matrix_sync(res, c_frag, 16, wmma::mem_row_major);

  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
}

template <class T, class R> float tensor_lat() {

  intilizeDeviceProp(0);

  THREADS_PER_BLOCK = 1;
  THREADS_PER_SM = 1;
  BLOCKS_NUM = 1;
  TOTAL_THREADS = 1;

  uint64_t *startClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  T *data1 = (T *)malloc(M_SIZE * sizeof(T));
  T *data2 = (T *)malloc(M_SIZE * sizeof(T));
  R *res = (R *)malloc(TOTAL_THREADS * sizeof(R));

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  T *data1_g;
  T *data2_g;
  R *res_g;

  for (uint32_t i = 0; i < M_SIZE; i++) {
    data1[i] = (T)i;
    data2[i] = (T)i;
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&data1_g, M_SIZE * sizeof(T)));
  gpuErrchk(cudaMalloc(&data2_g, M_SIZE * sizeof(T)));
  gpuErrchk(cudaMalloc(&res_g, TOTAL_THREADS * sizeof(R)));

  gpuErrchk(
      cudaMemcpy(data1_g, data1, M_SIZE * sizeof(T), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(data2_g, data2, M_SIZE * sizeof(T), cudaMemcpyHostToDevice));

  tensor_latency<T, R><<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
      startClk_g, stopClk_g, data1_g, data2_g, res_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  // gpuErrchk( cudaMemcpy(res, res_g, M_SIZE*sizeof(R), cudaMemcpyDeviceToHost)
  // );

  float wmma, hmma;
  uint64_t total_time = stopClk[0] - startClk[0];
  wmma = ((float)(total_time)) / ((float)(REPEAT_ITERS));
  hmma =
      ((float)(total_time)) / ((float)(REPEAT_ITERS * SASS_hmma_per_PTX_wmma));

  std::cout << "wmma latency = " << wmma << "(clk)\n";
  std::cout << "hmma latency = " << hmma << "(clk)\n";
  std::cout << "Total Clk number = " << total_time << "\n";

  return wmma;
}

#endif