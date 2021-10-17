
#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"
#include "../MaxFlops_int32/MaxFlops_int32.h"
#include "../lat_int32/lat_int32.h"

int main() {
  intilizeDeviceProp(0);

  float flops = max_int32_flops();
  float latency = int32_latency();

  if (ACCEL_SIM_MODE) {
    unsigned lat = (unsigned)latency;
    // divide by 2 as we need FMA thoughput
    unsigned throughput_per_SM = round_up_2n(flops / 2);
    float throughput_per_sched = (float)throughput_per_SM / WARP_SCHEDS_PER_SM;

    unsigned init = WARP_SIZE / throughput_per_sched;

    std::cout << "\n//Accel_Sim config: \n";
    if (deviceProp.major < 6) { // detecaited integer unit was added since Volta
      std::cout << "-gpgpu_num_int_units 0" << std::endl;
    } else {
      std::cout << "-gpgpu_num_int_units " << WARP_SCHEDS_PER_SM << std::endl;
      std::cout << "-ptx_opcode_latency_int " << lat << "," << lat << "," << lat
                << "," << lat << ",21" << std::endl;
      std::cout << "-ptx_opcode_initiation_int " << init << "," << init << ","
                << init << "," << init << "," << init << std::endl;
      std::cout << "-trace_opcode_latency_initiation_int " << lat << "," << init
                << std::endl;
    }
  }

  return 1;
}