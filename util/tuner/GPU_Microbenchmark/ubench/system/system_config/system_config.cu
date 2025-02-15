#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"

int main() {
  intilizeDeviceProp(0);

  printf("Device Name = %s\n", deviceProp.name);
  printf("GPU Max Clock rate = %.0f MHz \n", deviceProp.clockRate * 1e-3f);
  printf("GPU Base Clock rate = %d MHz \n", CLK_FREQUENCY);
  printf("SM Count = %d\n", deviceProp.multiProcessorCount);
  printf("CUDA version number = %d.%d\n", deviceProp.major, deviceProp.minor);

  if (ACCEL_SIM_MODE) {

    std::cout << "\n//Accel_Sim config: \n";

    float mem_freq_MHZ = (deviceProp.memoryClockRate * 1e-3f * 2) /
                         dram_model_freq_ratio[DRAM_MODEL];
    std::cout << "-gpgpu_compute_capability_major " << deviceProp.major
              << std::endl;
    std::cout << "-gpgpu_compute_capability_minor " << deviceProp.minor
              << std::endl;
    std::cout << "-gpgpu_n_clusters " << deviceProp.multiProcessorCount
              << std::endl;
    std::cout << "-gpgpu_n_cores_per_cluster 1" << std::endl;
    std::cout << "-gpgpu_clock_domains " << CLK_FREQUENCY << ":"
              << CLK_FREQUENCY << ":" << CLK_FREQUENCY << ":" << mem_freq_MHZ
              << std::endl;
  }

  return 1;
}
