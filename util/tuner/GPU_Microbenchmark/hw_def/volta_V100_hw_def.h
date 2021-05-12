// These are the configration parameters that can be found publicly
// Volta QV100 HW def file (sm_70)
// Data source:
// https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf

#ifndef VOLTA_V100_HW_DEF_H
#define VOLTA_V100_HW_DEF_H

#include "./common/common.h"
#include "./common/deviceQuery.h"

#define L1_SIZE (128 * 1024) // Max L1 size in bytes

#define CLK_FREQUENCY 1455 // frequency in MHz

#define ISSUE_MODEL issue_model::single
#define CORE_MODEL core_model::subcore
#define DRAM_MODEL dram_model::HBM
#define WARP_SCHEDS_PER_SM 4

// see slide 22 at
// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21730-inside-the-nvidia-ampere-architecture.pdf
// number of SASS HMMA per 16x16 PTX WMMA for FP16 - FP32 accumlate operation
#define SASS_hmma_per_PTX_wmma  16 

// These vars are almost constant between HW generations
// see slide 24 at
// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21730-inside-the-nvidia-ampere-architecture.pdf
// each memory channel is supported by 2 L2 banks
#define L2_BANKS_PER_MEM_CHANNEL 2 
#define L2_BANK_WIDTH_in_BYTE 32

#endif