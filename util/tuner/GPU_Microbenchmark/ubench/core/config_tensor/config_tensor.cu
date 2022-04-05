#include "../tensor_bw_half/tensor_bw_half.h"
#include "../tensor_lat_half/tensor_lat_half.h"

int main() {
  intilizeDeviceProp(0);

  // measure the flops and lat based on half operand and float accumlate
  float flops = tensor_max_flops<half, float>();
  float latency = tensor_lat<half, float>();

  if (ACCEL_SIM_MODE) {
    unsigned lat = (unsigned)latency;
    unsigned throughput_per_SM = round_up_2n(flops);
    float throughput_per_sched = (float)throughput_per_SM / WARP_SCHEDS_PER_SM;

    unsigned init = WARP_SIZE / throughput_per_sched;

    std::cout << "\n//Accel_Sim config: \n";
    if (deviceProp.major < 6) { // tensor core was added since Volta
      std::cout << "-gpgpu_tensor_core_avail 0" << std::endl;
      std::cout << "-gpgpu_num_tensor_core_units 0" << std::endl;
    } else {
      std::cout << "-gpgpu_tensor_core_avail 1" << std::endl;
      std::cout << "-gpgpu_num_tensor_core_units " << WARP_SCHEDS_PER_SM
                << std::endl;
      std::cout << "-ptx_opcode_latency_tesnor " << lat << std::endl;
      std::cout << "-ptx_opcode_initiation_tensor " << init << std::endl;

      // trace mode
      // assume tesnor is on spec unit 3
      std::cout << "-trace_opcode_latency_initiation_tensor "
                << lat / SASS_hmma_per_PTX_wmma << ","
                << init / SASS_hmma_per_PTX_wmma << std::endl;
      std::cout << "-specialized_unit_3 1," << WARP_SCHEDS_PER_SM << ","
                << lat / SASS_hmma_per_PTX_wmma << "," << WARP_SCHEDS_PER_SM
                << "," << WARP_SCHEDS_PER_SM << ",TENSOR" << std::endl;
      std::cout << "-trace_opcode_latency_initiation_spec_op_3 "
                << lat / SASS_hmma_per_PTX_wmma << ","
                << init / SASS_hmma_per_PTX_wmma << std::endl;
    }
  }

  return 1;
}