// These are the configration parameters that can be found publicly sources
// Turing HW def file
#ifndef TURING_RTX2070_DEF_H
#define TURING_RTX2070_DEF_H

#include "./common/common.h"
#include "./common/deviceQuery.h"

#define L1_SIZE (64 * 1024) // Max L1 size in bytes

#define CLK_FREQUENCY 1365 // frequency in MHz

#define ISSUE_MODEL issue_model::single   // single issue core or dual issue
#define CORE_MODEL core_model::subcore    // subcore model or shared model
#define DRAM_MODEL dram_model::GDDR6      // memory type
#define WARP_SCHEDS_PER_SM 4              // number of warp schedulers per SM

// number of SASS HMMA per 16x16 PTX WMMA for FP16 - FP32 accumlate operation
#define SASS_hmma_per_PTX_wmma 4 

// These vars are almost constant between HW generation
// see slide 24 from Nvidia at
// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21730-inside-the-nvidia-ampere-architecture.pdf
#define L2_BANKS_PER_MEM_CHANNEL 2
#define L2_BANK_WIDTH_in_BYTE 32

#endif
