#ifndef WATCHDOG_H
#define WATCHDOG_H

#include <assert.h>
#include <stdint.h>
#include <string.h>

/* nvbit interface file for Instr class */
#include "nvbit.h"

/** 
 * Use this struct during instrumentation to skip instructions between
 * WARPSYNC.COLLECTIVE and its target instruction (inclusive) 
 * */
struct WarpsyncCollectiveWatchdog{
  bool in_region = false; // true if we are in the region of interest
  uint64_t start_pc = 0; // PC of the WARPSYNC.COLLECTIVE instruction 
  uint64_t end_pc = 0; // PC of the target instruction 

  void reset(){
    in_region = false;
    start_pc = 0;
    end_pc = 0;
  }

  bool is_in_region() const{
    return in_region;
  }

  void observe_instruction(Instr * const instr){
    if(in_region){
      if(instr->getOffset() > end_pc){
        reset();
      }
    }
    
    // if opcode is WARPSYNC.COLLECTIVE, set in_region to true
    if(strcmp(instr->getOpcode(), "WARPSYNC.COLLECTIVE") == 0){
      assert(1 < instr->getNumOperands());
      in_region = true;
      start_pc = instr->getOffset();
      end_pc = instr->getOperand(1)->u.imm_uint64.value;
    }
  }
};

#endif // WATCHDOG_H
