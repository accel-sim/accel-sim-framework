#include "../../l2_cache/l2_lat/l2_lat.h"
#include "mem_lat.h"
#include <iostream>

int main() {

  intilizeDeviceProp(0);

  float lat_mem = mem_lat();

  if (ACCEL_SIM_MODE) {
    float lat2 = l2_hit_lat();

    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-dram_latency " << (unsigned)(lat_mem - lat2) << std::endl;
  }

  return 1;
}
