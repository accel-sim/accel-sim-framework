#include "../../l1_cache/l1_lat/l1_lat.h"
#include "l2_lat.h"

int main() {

  intilizeDeviceProp(0);

  float lat2 = l2_hit_lat();

  if (ACCEL_SIM_MODE) {
    float lat1 = l1_lat();

    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-gpgpu_l2_rop_latency " << (unsigned)(lat2 - lat1)
              << std::endl;
  }

  return 1;
}
