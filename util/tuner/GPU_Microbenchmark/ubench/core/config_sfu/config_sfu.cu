
#include "../../../hw_def/hw_def.h"
#include "../sfu_bw_fsqrt/sfu_bw_fsqrt.h"
#include "../sfu_lat_fsqrt/sfu_lat_fsqrt.h"

int main() {
  intilizeDeviceProp(0);

  float flops = sfu_max_flops();
  float latency = sfu_latency();

  if (ACCEL_SIM_MODE) {
    unsigned lat = (unsigned)latency;
    unsigned throughput_per_SM = round_up_2n(flops);
    float throughput_per_sched = (float)throughput_per_SM / WARP_SCHEDS_PER_SM;

    unsigned init = WARP_SIZE / throughput_per_sched;

    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-gpgpu_num_sfu_units " << WARP_SCHEDS_PER_SM << std::endl;
    std::cout << "-ptx_opcode_latency_sfu " << lat << std::endl;
    std::cout << "-ptx_opcode_initiation_sfu " << init << std::endl;
    std::cout << "-trace_opcode_latency_initiation_sfu " << lat << "," << init
              << std::endl;
  }

  return 1;
}