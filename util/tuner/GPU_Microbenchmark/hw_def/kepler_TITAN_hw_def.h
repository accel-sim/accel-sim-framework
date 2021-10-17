// Kepler TITAN HW def file
#ifndef KEPLER_TITAN_DEF_H
#define KEPLER_TITAN_DEF_H

#include "./common/common.h"
#include "./common/deviceQuery.h"

#define L1_SIZE (64 * 1024) // Max L1 size in bytes, when enabled

#define CLK_FREQUENCY 837 // frequency in MHz

#define ISSUE_MODEL issue_model::dual
#define CORE_MODEL core_model::shared
#define DRAM_MODEL dram_model::GDDR5

#define WARP_SCHEDS_PER_SM 4

// no tensor cores in kepler
#define SASS_hmma_per_PTX_wmma 0

// These vars are almost constant between HW generations
#define L2_BANKS_PER_MEM_CHANNEL 2
#define L2_BANK_WIDTH_in_BYTE 32

#endif
