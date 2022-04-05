#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"

int main() {
  intilizeDeviceProp(0);

  printf("CUDA version number = %d.%d\n", deviceProp.major, deviceProp.minor);

  if (ACCEL_SIM_MODE) {
    std::cout << "\n//Accel_Sim config: \n";

    std::cout << "-gpgpu_ptx_force_max_capability " << deviceProp.major
              << deviceProp.minor << std::endl;

    std::cout << "-gpgpu_shader_registers " << deviceProp.regsPerMultiprocessor
              << std::endl;
    std::cout << "-gpgpu_registers_per_block " << deviceProp.regsPerBlock
              << std::endl;
    std::cout << "-gpgpu_occupancy_sm_number " << deviceProp.major
              << deviceProp.minor << std::endl;
    std::cout << "-gpgpu_coalesce_arch " << deviceProp.major << deviceProp.minor
              << std::endl;

    unsigned ID_OC_SP, ID_OC_DP, ID_OC_INT, ID_OC_SFU, ID_OC_MEM, OC_EX_SP,
        OC_EX_DP, OC_EX_INT, OC_EX_SFU, OC_EX_MEM, EX_WB, ID_OC_TENSOR_CORE,
        OC_EX_TENSOR_CORE;
    ID_OC_SFU = OC_EX_SFU = WARP_SCHEDS_PER_SM;
    ID_OC_MEM = OC_EX_MEM = WARP_SCHEDS_PER_SM;
    ID_OC_SP = OC_EX_SP = WARP_SCHEDS_PER_SM;
    ID_OC_DP = OC_EX_DP = WARP_SCHEDS_PER_SM;
    EX_WB = WARP_SCHEDS_PER_SM * 2;
    if (deviceProp.major < 6) { // no integer or tensor cores before volta
      ID_OC_INT = OC_EX_INT = 0;
      ID_OC_TENSOR_CORE = OC_EX_TENSOR_CORE = 0;
    } else {
      ID_OC_INT = OC_EX_INT = WARP_SCHEDS_PER_SM;
      ID_OC_TENSOR_CORE = OC_EX_TENSOR_CORE = WARP_SCHEDS_PER_SM;
    }

    //#
    // ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE
    std::cout << "-gpgpu_pipeline_widths " << ID_OC_SP << "," << ID_OC_DP << ","
              << ID_OC_INT << "," << ID_OC_SFU << "," << ID_OC_MEM << ","
              << OC_EX_SP << "," << OC_EX_DP << "," << OC_EX_INT << ","
              << OC_EX_SFU << "," << OC_EX_MEM << "," << EX_WB;
    if (deviceProp.major < 6)
      std::cout << std::endl;
    else
      std::cout << "," << ID_OC_TENSOR_CORE << "," << OC_EX_TENSOR_CORE
                << std::endl;

    std::cout << "-gpgpu_sub_core_model " << CORE_MODEL << std::endl;

    std::cout << "-gpgpu_enable_specialized_operand_collector 0" << std::endl;
    std::cout << "-gpgpu_operand_collector_num_units_gen "
              << WARP_SCHEDS_PER_SM * 2 << std::endl;
    std::cout << "-gpgpu_operand_collector_num_in_ports_gen "
              << WARP_SCHEDS_PER_SM * 2 << std::endl;
    std::cout << "-gpgpu_operand_collector_num_out_ports_gen "
              << WARP_SCHEDS_PER_SM * 2 << std::endl;

    std::cout << "-gpgpu_num_sched_per_core " << WARP_SCHEDS_PER_SM
              << std::endl;

    std::cout << "-gpgpu_max_insn_issue_per_warp " << ISSUE_MODEL << std::endl;
    std::cout << "-gpgpu_dual_issue_diff_exec_units " << (deviceProp.major > 3)
              << std::endl;

    std::cout << "-gpgpu_inst_fetch_throughput " << WARP_SCHEDS_PER_SM
              << std::endl;

    std::cout << "-gpgpu_shader_core_pipeline "
              << deviceProp.maxThreadsPerMultiProcessor << ":"
              << deviceProp.warpSize << std::endl;
    std::cout << "-gpgpu_shader_cta "
              << round_up_2n((unsigned)deviceProp.maxThreadsPerMultiProcessor /
                             deviceProp.warpSize / 2)
              << std::endl;
  }

  return 1;
}