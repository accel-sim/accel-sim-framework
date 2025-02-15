
#include "../../../hw_def/hw_def.h"
#include "../MaxFlops_double/MaxFlops_double.h"
#include "../lat_double/lat_double.h"

int main() {
  intilizeDeviceProp(0);

  float flops = dpu_max_flops();
  float latency = dpu_latency();

  if (ACCEL_SIM_MODE) {
    unsigned lat = (unsigned)latency;
    // divide flops by 2 as we need FMA throughput
    unsigned throughput_per_SM = round_up_2n(flops / 2);
    float throughput_per_sched = (float)throughput_per_SM / WARP_SCHEDS_PER_SM;

    unsigned init = WARP_SIZE / throughput_per_sched;

    //init cannot be larger than latency
    if(init > latency)
      latency = init;

    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-gpgpu_num_dp_units " << WARP_SCHEDS_PER_SM << std::endl;
    std::cout << "-ptx_opcode_latency_dp " << lat << "," << lat << "," << lat
              << "," << lat << ",330" << std::endl;
    std::cout << "-ptx_opcode_initiation_dp " << init << "," << init << ","
              << init << "," << init << ",130" << std::endl;
    std::cout << "-trace_opcode_latency_initiation_dp " << lat << "," << init
              << std::endl;
  }

  return 1;
}
