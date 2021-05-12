// This benchmark measures the maximum read bandwidth of GPU memory
// Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt mem_bw.cu

// This code have been tested on Volta V100 architecture
// You can check the mem BW from the nvprof and nvsight
// (dram_read_throughput+dram_write_throughput)

// to run the program with nvsight
// make nvsight ./mem_bw

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

/*
Send as many as float4 read requests on the flight to increase DRAM row buffer
locality and hit the max BW
*/

__global__ void mem_bw(float *A, float *B, float *C, float *D, float *E,
                       float *F, uint32_t *startClk, uint32_t *stopClk,
                       unsigned ARRAY_SIZE) {
  // block and thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  for (int i = idx; i < ARRAY_SIZE / 4; i += blockDim.x * gridDim.x) {
    float4 a1 = reinterpret_cast<float4 *>(A)[i];
    float4 b1 = reinterpret_cast<float4 *>(B)[i];
    float4 d1 = reinterpret_cast<float4 *>(D)[i];
    float4 e1 = reinterpret_cast<float4 *>(E)[i];
    float4 f1 = reinterpret_cast<float4 *>(F)[i];
    float4 c1;

    c1.x = a1.x + b1.x + d1.x + e1.x + f1.x;
    c1.y = a1.y + b1.y + d1.y + e1.y + f1.y;
    c1.z = a1.z + b1.z + d1.z + e1.z + f1.z;
    c1.w = a1.w + b1.w + d1.w + e1.w + f1.w;

    reinterpret_cast<float4 *>(C)[i] = c1;
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  // write time and data back to memory
  startClk[idx] = start;
  stopClk[idx] = stop;
}

int main() {
  intilizeDeviceProp(0);

  // Array size has to exceed L2 size to avoid L2 cache residence
  unsigned ARRAY_SIZE = (L2_SIZE / sizeof(float)) * 2;

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  float *A = (float *)malloc(ARRAY_SIZE * sizeof(float));
  float *B = (float *)malloc(ARRAY_SIZE * sizeof(float));
  float *C = (float *)malloc(ARRAY_SIZE * sizeof(float));
  float *D = (float *)malloc(ARRAY_SIZE * sizeof(float));
  float *E = (float *)malloc(ARRAY_SIZE * sizeof(float));
  float *F = (float *)malloc(ARRAY_SIZE * sizeof(float));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  float *A_g;
  float *B_g;
  float *C_g;
  float *D_g;
  float *E_g;
  float *F_g;

  for (uint32_t i = 0; i < ARRAY_SIZE; i++) {
    A[i] = (float)i;
    B[i] = (float)i;
    D[i] = (float)i;
    E[i] = (float)i;
    F[i] = (float)i;
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&A_g, ARRAY_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc(&B_g, ARRAY_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc(&C_g, ARRAY_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc(&D_g, ARRAY_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc(&E_g, ARRAY_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc(&F_g, ARRAY_SIZE * sizeof(float)));

  gpuErrchk(
      cudaMemcpy(A_g, A, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(B_g, B, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(D_g, D, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(E_g, E, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(F_g, F, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  mem_bw<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(A_g, B_g, C_g, D_g, E_g, F_g,
                                            startClk_g, stopClk_g, ARRAY_SIZE);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(C, C_g, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

  float mem_bw;
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  unsigned N = ARRAY_SIZE * 6 * sizeof(float); // 6 arrays of floats types
  float max_bw = (float)MEM_BITWIDTH * MEM_CLK_FREQUENCY * 2 / 1e3 / 8;
  mem_bw = (float)(N) / ((float)(stopClk[0] - startClk[0]));
  printf("Mem BW= %f (Byte/Clk)\n", mem_bw);
  printf("Mem BW= %f (GB/sec)\n", (float)N / milliseconds / 1e6);
  printf("Max Theortical Mem BW= %f (GB/sec)\n", max_bw);
  printf("Mem Efficiency = %f %%\n", (mem_bw / max_bw) * 100);

  printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);
}
