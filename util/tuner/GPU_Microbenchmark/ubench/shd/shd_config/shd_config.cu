#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"

int main() {
  intilizeDeviceProp(0);

  printf("Shared memory per multiprocessor = %lu bytes\n",
         deviceProp.sharedMemPerMultiprocessor);

  printf("Shared memory per block = %lu bytes\n", deviceProp.sharedMemPerBlock);

  if (ACCEL_SIM_MODE) {

    std::cout << "\n//Accel_Sim config: \n";

    std::cout << "-gpgpu_shmem_size " << deviceProp.sharedMemPerMultiprocessor
              << std::endl;
    std::cout << "-gpgpu_shmem_sizeDefault "
              << deviceProp.sharedMemPerMultiprocessor << std::endl;
    std::cout << "-gpgpu_shmem_per_block " << deviceProp.sharedMemPerBlock
              << std::endl;
  }

  return 1;
}