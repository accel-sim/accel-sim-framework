// Pascal ITIANX HW def file
#ifndef PASCAL_TITANX_DEF_H
#define PASCAL_TITANX_DEF_H

#include "./common/common.h"
#include "./common/deviceQuery.h"

#define L1_SIZE (24 * 1024) // Max L1 size in bytes, when enabled

#define CLK_FREQUENCY 1417 // frequency in MHz

#define ISSUE_MODEL issue_model::dual
#define CORE_MODEL core_model::subcore
#define DRAM_MODEL dram_model::GDDR5X
#define WARP_SCHEDS_PER_SM 4

// no tensor cores in pascal
#define SASS_hmma_per_PTX_wmma 0

// These vars are almost constant between HW generations
#define L2_BANKS_PER_MEM_CHANNEL 2
#define L2_BANK_WIDTH_in_BYTE 32

#endif
