#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"

int main() {
  intilizeDeviceProp(0);

  char msg[256];
  snprintf(msg, sizeof(msg), "Global memory size = %.0f GB\n",
           static_cast<float>(deviceProp.totalGlobalMem / 1073741824.0f));
  std::cout << msg;
  std::cout << "Memory Clock rate = " << deviceProp.memoryClockRate * 1e-3f
            << " Mhz\n";
  std::cout << "Memory Bus Width = " << deviceProp.memoryBusWidth << " bit\n";
  std::cout << "Memory type = " << dram_model_str[DRAM_MODEL] << "\n";
  std::cout << "Memory channels = "
            << get_num_channels(deviceProp.memoryBusWidth, DRAM_MODEL) << "\n";

  if (ACCEL_SIM_MODE) {

    std::cout << "\n//Accel_Sim config: \n";

    std::cout << "-gpgpu_n_mem "
              << get_num_channels(deviceProp.memoryBusWidth, DRAM_MODEL)
              << std::endl;

    std::cout << "-gpgpu_n_mem_per_ctrlr "
              << dram_model_mem_per_ctrlr[DRAM_MODEL] << std::endl;
    std::cout << "-gpgpu_dram_buswidth " << dram_model_bus_width[DRAM_MODEL] / 8
              << std::endl;
    std::cout << "-gpgpu_dram_burst_length "
              << dram_model_burst_length[DRAM_MODEL] << std::endl;
    std::cout << "-dram_data_command_freq_ratio "
              << dram_model_freq_ratio[DRAM_MODEL] << std::endl;

    // timing
    float device_freq_MHZ = (deviceProp.memoryClockRate * 1e-3f * 2) /
                            dram_model_freq_ratio[DRAM_MODEL];
    if (DRAM_MODEL == dram_model::HBM) {
      // use HBM timing
      DDR_Timing timing = HBM_Timing_1000MHZ;
      timing.scale_timing_for_new_freq(device_freq_MHZ);
      std::cout << "-dram_dual_bus_interface 1" << std::endl;
      std::cout << "-gpgpu_dram_timing_opt nbk=" << timing.nbk
                << ":CCD=" << get_adjusted_CCD(DRAM_MODEL)
                << ":RRD=" << timing.RRD << ":RCD=" << timing.RCD
                << ":RAS=" << timing.RAS << ":RP=" << timing.RP
                << ":RC=" << timing.RC << ":CL=" << timing.CL
                << ":WL=" << timing.WL << ":CDLR=" << timing.CDLR
                << ":WR=" << timing.WR << ":nbkgrp=" << timing.nbkgrp
                << ":CCDL=" << timing.CCDL << ":RTPL=" << timing.RTPL
                << std::endl;
    } else {
      // use GDDR timing
      DDR_Timing timing = GDDR5_Timing_1800MHZ;
      timing.scale_timing_for_new_freq(device_freq_MHZ);
      std::cout << "-dram_dual_bus_interface 0" << std::endl;
      std::cout << "-gpgpu_dram_timing_opt nbk=" << timing.nbk
                << ":CCD=" << get_adjusted_CCD(DRAM_MODEL)
                << ":RRD=" << timing.RRD << ":RCD=" << timing.RCD
                << ":RAS=" << timing.RAS << ":RP=" << timing.RP
                << ":RC=" << timing.RC << ":CL=" << timing.CL
                << ":WL=" << timing.WL << ":CDLR=" << timing.CDLR
                << ":WR=" << timing.WR << ":nbkgrp=" << timing.nbkgrp
                << ":CCDL=" << timing.CCDL << ":RTPL=" << timing.RTPL
                << std::endl;
    }

    // leave the adddress mapping for now as it is
    // the number of banks in HBM and GDDR are 16 and atom size is 32B, so the
    // mapping should be okay. TODO: make this to be varibale based on memory
    // model and size std::cout<<"-gpgpu_mem_address_mask 1"<<std::endl;
    // std::cout<<"-gpgpu_mem_addr_mapping
    // dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS"<<std::endl;
  }

  return 1;
}