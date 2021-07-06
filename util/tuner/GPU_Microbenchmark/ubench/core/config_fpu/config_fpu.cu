
#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"
#include "../MaxFlops_float/MaxFlops_float.h"
#include "../lat_float/lat_float.h"

int main() {
  intilizeDeviceProp(0);

  float flops = fpu_max_flops();
  float latency = fpu_latency();

  if (ACCEL_SIM_MODE) {
    unsigned lat = (unsigned)latency;
    // divide flops by 2 as we need FMA thoughput
    unsigned throughput_per_SM = round_up_2n(flops / 2);
    float throughput_per_sched = (float)throughput_per_SM / WARP_SCHEDS_PER_SM;

    unsigned init = WARP_SIZE / throughput_per_sched;

    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-gpgpu_num_sp_units " << WARP_SCHEDS_PER_SM << std::endl;
    std::cout << "-ptx_opcode_latency_fp " << lat << "," << lat << "," << lat
              << "," << lat << ",39" << std::endl;
    std::cout << "-ptx_opcode_initiation_fp " << init << "," << init << ","
              << init << "," << init << "," << init * 2 << std::endl;
    std::cout << "-trace_opcode_latency_initiation_sp " << lat << "," << init
              << std::endl;
  }

  return 1;
}