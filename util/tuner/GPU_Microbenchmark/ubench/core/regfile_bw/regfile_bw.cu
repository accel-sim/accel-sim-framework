#include "../MaxFlops_float/MaxFlops_float.h"
#include "../tensor_bw_half/tensor_bw_half.h"

int main() {
  intilizeDeviceProp(0);

  unsigned regfile_bw;
  /* we measure the reg file BW based on the most demanding data instruction,
   i.e. tensor cores. See slide 20 from Nvidia for more details at
   https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21730-inside-the-nvidia-ampere-architecture.pdf
  */
  if (deviceProp.major >= 6) { // tesnor core unit was added since Volta
    float fma_bw = tensor_max_flops<half, float>(true);

    unsigned tensor_MACs_per_SM = round_up_2n(fma_bw);

    /*
    two operands needs per MAC each cycle (A, B), C will be saved at the tensor
    core accuamlte register
    */
    regfile_bw = tensor_MACs_per_SM * sizeof(half) * 2;
  } else {
    // if less than volta calculate based on FP32 FMA
    float flops = fpu_max_flops();

    // divide by 2 as we need FMA throughput
    unsigned FMA_throughput_per_SM = round_up_2n(flops / 2);

    // three operands needs per FMA each cycle (A, B, C)
    regfile_bw = round_up_2n((float)FMA_throughput_per_SM * sizeof(float) * 3);
  }
  std::cout << "\nregfile_bw = " << regfile_bw << " (byte/SM)" << std::endl;

  if (ACCEL_SIM_MODE) {

    unsigned reg_ports;
    // Nvidia starts to have dual port register file since volta
    if (deviceProp.major < 6)
      reg_ports = 1;
    else
      reg_ports = 2;

    // WARP_SIZE*4 bytes, as registers are 32-bit width
    unsigned banks_num = regfile_bw / (WARP_SIZE * 4) / reg_ports;

    /* we multiply by two as accel-sim does not model register file cache (added
     since kepler) so to mitigate, the reg file bw comes from RFC, we
     conservatively multiply the banks by 2 (to fix)
     */
    if (deviceProp.major > 3)
      banks_num = banks_num * 2;

    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-gpgpu_num_reg_banks " << banks_num << std::endl;
    std::cout << "-gpgpu_reg_file_port_throughput " << reg_ports << std::endl;
  }

  return 1;
}
