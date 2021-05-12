#include "l1_lat.h"

int main() {

  intilizeDeviceProp(0);

  float lat = l1_lat();

  if (ACCEL_SIM_MODE) {
    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-gpgpu_l1_latency " << (unsigned)lat << std::endl;
  }

  return 1;
}
