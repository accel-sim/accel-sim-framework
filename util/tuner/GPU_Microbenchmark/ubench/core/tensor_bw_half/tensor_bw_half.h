#ifndef MAXFLOPS_TENSOR_DEF_H
#define MAXFLOPS_TENSOR_DEF_H

#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define REPEAT_TIMES 2048
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define A_SIZE WMMA_M *WMMA_K
#define B_SIZE WMMA_N *WMMA_K
#define R_SIZE WMMA_M *WMMA_N

using namespace nvcuda;

template <class T, class R>
__global__ void
max_flops(uint64_t *startClk, uint64_t *stopClk, T *a, T *b, R *res,
          uint32_t strid) { // strid set to 0 used to prevent optimization

  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + tid;
  uint32_t warpid = gid / warpSize;

  a = a + warpid * A_SIZE;
  b = b + warpid * B_SIZE;
  res = res + warpid * R_SIZE;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, R> c_frag;

  wmma::load_matrix_sync(a_frag, a, 16);
  wmma::fill_fragment(c_frag, 0.0f);
  wmma::load_matrix_sync(b_frag, b, 16);

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  for (int j = 0; j < REPEAT_TIMES; ++j) {
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  wmma::store_matrix_sync(res, c_frag, WMMA_N, wmma::mem_row_major);

  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
}

template <class T, class R> float tensor_max_flops(bool report_fma_bw = false) {
  intilizeDeviceProp(0);

  BLOCKS_NUM = 1;
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  unsigned total_A_SIZE =
      A_SIZE * (TOTAL_THREADS / WARP_SIZE); // asume one 16x16 matrix per warp
  unsigned total_B_SIZE =
      B_SIZE * (TOTAL_THREADS / WARP_SIZE); // asume one 16x16 matrix per warp
  unsigned total_R_SIZE =
      R_SIZE * (TOTAL_THREADS / WARP_SIZE); // asume one 16x16 matrix per warp

  uint64_t *startClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  T *data1 = (T *)malloc(total_A_SIZE * sizeof(T));
  T *data2 = (T *)malloc(total_B_SIZE * sizeof(T));
  R *res = (R *)malloc(total_R_SIZE * sizeof(R));

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  T *data1_g;
  T *data2_g;
  R *res_g;

  for (uint32_t i = 0; i < A_SIZE; i++) {
    data1[i] = (T)i;
  }

  for (uint32_t i = 0; i < B_SIZE; i++) {
    data2[i] = (T)i;
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&data1_g, total_A_SIZE * sizeof(T)));
  gpuErrchk(cudaMalloc(&data2_g, total_B_SIZE * sizeof(T)));
  gpuErrchk(cudaMalloc(&res_g, total_R_SIZE * sizeof(R)));

  gpuErrchk(cudaMemcpy(data1_g, data1, total_A_SIZE * sizeof(T),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(data2_g, data2, total_B_SIZE * sizeof(T),
                       cudaMemcpyHostToDevice));

  max_flops<T, R><<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
      startClk_g, stopClk_g, data1_g, data2_g, res_g, 0);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(res, res_g, total_R_SIZE * sizeof(R), cudaMemcpyDeviceToHost));

  float wmma_bw, hmma_bw, fma_bw;
  uint64_t total_time =
      *std::max_element(&stopClk[0], &stopClk[TOTAL_THREADS]) -
      *std::min_element(&startClk[0], &startClk[TOTAL_THREADS]);
  wmma_bw = ((float)(REPEAT_TIMES * TOTAL_THREADS)) / (float)total_time;
  hmma_bw = ((float)(REPEAT_TIMES * TOTAL_THREADS * SASS_hmma_per_PTX_wmma)) /
            (float)total_time;
  fma_bw = ((float)(REPEAT_TIMES * WMMA_M * WMMA_N * WMMA_K *
                    (TOTAL_THREADS / WARP_SIZE))) /
           (float)total_time;

  std::cout << "wmma PTX issue bandwidth = " << wmma_bw << "(thread/clk/SM) \n";
  std::cout << "hmma SASS issue bandwidth = " << hmma_bw << "(thread/clk/SM)\n";
  std::cout << "FMA tensor bandwidth = " << fma_bw << "(FMA/clk/SM)\n";

  std::cout << "Total Clk number = " << total_time << "\n";

  if (report_fma_bw)
    return fma_bw;
  else
    return wmma_bw;
}

#endif
