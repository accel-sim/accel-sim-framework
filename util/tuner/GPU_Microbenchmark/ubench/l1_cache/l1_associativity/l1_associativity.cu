#include <cuda.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include "../../../hw_def/hw_def.h"

class chaserParam {
public:
  uint32_t stride, array_size, iteration, l1_cache_size;
  int shared_mem_size_byte;
  bool sequential;

  uint64_t start, stop;
};

__global__ void setup_kernel(curandStateMRG32k3a *state) {
  int id = 0;

  curand_init(1234, id, 0, &state[id]);
}

__global__ void l1_squential(uint64_t *startCLK, uint64_t *stopCLK,
                             uint32_t *dsink, uint32_t *posArray,
                             uint32_t stride, uint32_t array_size,
                             uint32_t iteration) {
  //    uint32_t tid = threadIdx.x;
  //    uint32_t bid = blockIdx.x;
  //    uint32_t uid = bid*blockDim.x+tid;
  //    uint32_t n_threads = blockDim.x * gridDim.x;

  uint64_t start, stop;
  uint32_t pointer;
  pointer = 0;

  for (int itr = 0; itr < iteration; itr++) {
    start = clock64();
    for (uint32_t i = 0; i < (array_size / stride); i++) {
      pointer = posArray[pointer];
    }
    stop = clock64();
  }

  startCLK[0] = start;
  stopCLK[0] = stop;
  dsink[0] = pointer;
}

__global__ void l1_random(uint64_t *startCLK, uint64_t *stopCLK,
                          uint32_t *dsink, uint32_t *posArray,
                          curandStateMRG32k3a *state, uint32_t stride,
                          uint32_t array_size, uint32_t iteration) {
  //    uint32_t tid = threadIdx.x;
  //    uint32_t bid = blockIdx.x;
  //    uint32_t uid = bid*blockDim.x+tid;
  //    uint32_t n_threads = blockDim.x * gridDim.x;
  uint64_t start, stop;
  uint32_t pointer;
  pointer = 0;

  for (int itr = 0; itr < iteration; itr++) {
    start = clock64();
    for (uint32_t i = 0; i < (array_size / stride); i++) {
      pointer =
          posArray[(pointer + curand(state)) % array_size / stride * stride];
    }
    stop = clock64();
  }

  startCLK[0] = start;
  stopCLK[0] = stop;
  dsink[0] = pointer;
}

void l1_structure(chaserParam &chaser) {

  uint64_t *startCLK = (uint64_t *)malloc(1 * sizeof(uint64_t));
  uint64_t *stopCLK = (uint64_t *)malloc(1 * sizeof(uint64_t));
  uint32_t *dsink = (uint32_t *)malloc(1 * sizeof(uint32_t));
  uint32_t *posArray = (uint32_t *)malloc(chaser.array_size * sizeof(uint32_t));

  for (uint32_t i = 0; i < chaser.array_size; i++)
    posArray[i] = (i + chaser.stride) % chaser.array_size;

  uint64_t *startCLK_g;
  uint64_t *stopCLK_g;
  uint32_t *dsink_g;
  uint32_t *posArray_g;

  gpuErrchk(cudaMalloc(&startCLK_g, 1 * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopCLK_g, 1 * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&dsink_g, 1 * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&posArray_g, chaser.array_size * sizeof(uint32_t)));

  gpuErrchk(cudaMemcpy(posArray_g, posArray,
                       chaser.array_size * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));

  if (chaser.sequential) {
    l1_squential<<<1, 1>>>(startCLK_g, stopCLK_g, dsink_g, posArray_g,
                           chaser.stride, chaser.array_size, chaser.iteration);
  } else {
    curandStateMRG32k3a *devMRGStates;
    gpuErrchk(cudaMalloc((void **)&devMRGStates, sizeof(curandStateMRG32k3a)));
    setup_kernel<<<1, 1>>>(devMRGStates);
    l1_random<<<1, 1>>>(startCLK_g, stopCLK_g, dsink_g, posArray_g,
                        devMRGStates, chaser.stride, chaser.array_size,
                        chaser.iteration);
  }

  // gpuErrchk( cudaPeekAtLastError() );

  gpuErrchk(cudaMemcpy(startCLK, startCLK_g, 1 * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopCLK, stopCLK_g, 1 * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(dsink, dsink_g, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  chaser.start = startCLK[0];
  chaser.stop = stopCLK[0];

  free(startCLK);
  free(stopCLK);
  free(dsink);
  free(posArray);
  gpuErrchk(cudaFree(startCLK_g));
  gpuErrchk(cudaFree(stopCLK_g));
  gpuErrchk(cudaFree(dsink_g));
  gpuErrchk(cudaFree(posArray_g));

  return;
}

int main() {
  intilizeDeviceProp(0);

  std::cout << "Launching L1 cache line size ubench" << std::endl;
  std::ostringstream oss;
  oss << "L1line.csv";
  std::string filename = oss.str();
  std::ofstream myfile1(filename);

  chaserParam chaser1;
  chaser1.shared_mem_size_byte = 0;
  chaser1.iteration = 1;
  chaser1.array_size = L1_SIZE / 4;
  chaser1.sequential = true;

  myfile1 << "chaser.stride,chaser.start,chaser.stop\n";
  for (uint32_t i = 1; i <= 32; i *= 2) {
    chaser1.stride = i;
    l1_structure(chaser1);
    myfile1 << chaser1.stride << "," << chaser1.start << "," << chaser1.stop
            << "\n";
  }

  std::cout << "Saving L1 cache line size data at L1line.csv" << std::endl;

  std::cout << "Launching L1 cache assoc ubench" << std::endl;
  std::ostringstream string;
  string << "L1asso.csv";
  filename = string.str();
  std::ofstream myfile2(filename);

  chaser1.iteration = 2;
  chaser1.sequential = false;
  // chaser1.array_size=L1_SIZE*8; //4096KB 32xl1size
  myfile2 << "chaser.stride,chaser.start,chaser.stop\n";
  for (uint32_t i = 8; i <= 128; i *= 2) {
    chaser1.stride = i;
    chaser1.array_size = L1_SIZE / 16 * i;
    l1_structure(chaser1);
    myfile2 << chaser1.stride << "," << chaser1.start << "," << chaser1.stop
            << "\n";
  }

  std::cout << "Saving L1 cache assoc data at L1asso.csv" << std::endl;
  myfile1.close();
  myfile2.close();
  return 1;
}
