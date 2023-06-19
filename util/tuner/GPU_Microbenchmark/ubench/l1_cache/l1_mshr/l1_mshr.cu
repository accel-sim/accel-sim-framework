#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "../../../hw_def/hw_def.h"

__global__ void l1_mshr(uint64_t *timing, uint32_t *dsink, uint32_t *posArray,
                        uint32_t stride, uint64_t array_size,
                        uint32_t iteration) {

  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;
  // uint32_t n_threads = blockDim.x * gridDim.x;
  // uint32_t record_length = MAX_SHARED_MEM_SIZE_PER_BLOCK/8;

  extern __shared__ uint32_t t_val[]; // size of shared memory

  uint32_t pointer;
  pointer = uid * 1024;
  asm volatile("bar.sync 0;");

  // pointer chasing
  for (int itr = 0; itr < iteration; itr++) {
    pointer = posArray[pointer];
    t_val[uid * iteration + itr] = clock64();
  }
  // pointer chasing completed
  for (uint32_t i = 0; i < iteration; i++) {
    timing[uid * iteration + i] = t_val[uid * iteration + i];
  }

  dsink[uid] = pointer;
}

void l1_structure(uint32_t stride, uint64_t array_size,
                  int shared_mem_size_byte, uint32_t iteration) {

  std::ostringstream oss;
  oss << "MSHR" << stride << "_array" << array_size << "_shmem"
      << (shared_mem_size_byte / 4) << "_itr" << iteration << ".csv";
  std::string filename = oss.str();
  std::ofstream myfile(filename);

  std::cout << "Launching L1 MSHR ubench" << std::endl;

  uint64_t *timing =
      (uint64_t *)malloc(TOTAL_THREADS * iteration * sizeof(uint64_t));
  uint32_t *dsink = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *posArray = (uint32_t *)malloc(array_size * sizeof(uint32_t));
  // uint32_t *val_array = (uint32_t*) malloc(array_size*sizeof(uint32_t));

  for (uint32_t i = 0; i < array_size; i++)
    posArray[i] = (i + stride) % array_size;

  uint64_t *timing_g;
  uint32_t *dsink_g;
  uint32_t *posArray_g;

  gpuErrchk(
      cudaMalloc(&timing_g, TOTAL_THREADS * iteration * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&dsink_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&posArray_g, array_size * sizeof(uint32_t)));

  gpuErrchk(cudaMemcpy(posArray_g, posArray, array_size * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));

  // cudaFuncSetAttribute(l1_mshr,
  // cudaFuncAttributePreferredSharedMemoryCarveout, 100); //set shared memory
  // size
  cudaFuncSetAttribute(l1_mshr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shared_mem_size_byte);
  l1_mshr<<<BLOCKS_NUM, THREADS_PER_BLOCK, shared_mem_size_byte>>>(
      timing_g, dsink_g, posArray_g, stride, array_size, iteration);

  // gpuErrchk( cudaPeekAtLastError() );

  gpuErrchk(cudaMemcpy(timing, timing_g,
                       TOTAL_THREADS * iteration * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(dsink, dsink_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));

  myfile << "thread_num,timing1,timing2,timing3,timing4,timing5,timing6\n";
  for (uint32_t thr = 0; thr < TOTAL_THREADS; thr += 32) {
    for (uint32_t itr = 0; itr < iteration; itr++) {
      if (itr != 0) {
        myfile << ",";

      } else {
        myfile << thr << ",";
      }
      myfile << timing[thr * iteration + itr];
    }
    myfile << "\n";
  }

  free(timing);
  free(dsink);
  free(posArray);
  gpuErrchk(cudaFree(timing_g));
  gpuErrchk(cudaFree(dsink_g));
  gpuErrchk(cudaFree(posArray_g));

  myfile.close();
  std::cout << "Saving L1 MSHR data at " << filename << std::endl;

  return;
}

int main() {
  intilizeDeviceProp(0);

  BLOCKS_NUM = 1;
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;
  THREADS_PER_SM = THREADS_PER_BLOCK * BLOCKS_NUM;

  uint32_t stride, iteration;
  int shared_mem_size_byte = MAX_SHARED_MEM_SIZE_PER_BLOCK;
  /*
          #ifdef VOLTA_HW_DEF_H
          uint32_t l1_cache_size = L1_SIZE-shared_mem_size_byte; //volta
     sharedmem is a partition of L1 #else uint32_t l1_cache_size = L1_SIZE;
          #endif
  */
  // measure line size and mshr
  stride = 100;
  iteration = 6;
  uint64_t array_size = 1024 * 1024 * 1024;
  l1_structure(stride, array_size, shared_mem_size_byte, iteration);

  /*
  //measure associativity
  stride = 8;
  iteration = 1;
  for (array_size=l1_cache_size/4; array_size<(l1_cache_size+512)/4;
  array_size++){
          //l1_structure (stride, array_size, shared_mem_size_byte, iteration);
  }
  */
  return 1;
}
