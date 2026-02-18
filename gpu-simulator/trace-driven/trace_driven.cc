// Copyright (c) 2018-2021, Mahmoud Khairy, Vijay Kandiah, Timothy Rogers, Tor
// M. Aamodt, Nikos Hardavellas
// Northwestern University, Purdue University, The University of British
// Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of Northwestern University, Purdue University,
//    The University of British Columbia nor the names of their contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../ISA_Def/accelwattch_component_mapping.h"
#include "../ISA_Def/ampere_opcode.h"
#include "../ISA_Def/hopper_opcode.h"
#include "../ISA_Def/kepler_opcode.h"
#include "../ISA_Def/pascal_opcode.h"
#include "../ISA_Def/trace_opcode.h"
#include "../ISA_Def/turing_opcode.h"
#include "../ISA_Def/volta_opcode.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu_context.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "trace_driven.h"

const trace_warp_inst_t *trace_shd_warp_t::get_next_trace_inst() {
  if (trace_pc < warp_traces.size()) {
    trace_warp_inst_t *new_inst =
        new trace_warp_inst_t(get_shader()->get_config());
    new_inst->parse_from_trace_struct(
        warp_traces[trace_pc], m_kernel_info->OpcodeMap,
        m_kernel_info->m_tconfig, m_kernel_info->m_kernel_trace_info);
    trace_pc++;
    return new_inst;
  } else
    return NULL;
}

void trace_shd_warp_t::clear() {
  trace_pc = 0;
  warp_traces.clear();
}

// functional_done
bool trace_shd_warp_t::trace_done() { return trace_pc == (warp_traces.size()); }

address_type trace_shd_warp_t::get_start_trace_pc() {
  assert(warp_traces.size() > 0);
  return warp_traces[0].m_pc;
}

address_type trace_shd_warp_t::get_pc() {
  assert(warp_traces.size() > 0);
  assert(trace_pc < warp_traces.size());
  return warp_traces[trace_pc].m_pc;
}

trace_kernel_info_t::trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                                         trace_function_info *m_function_info,
                                         trace_parser *parser,
                                         class trace_config *config,
                                         kernel_trace_t *kernel_trace_info)
    : kernel_info_t(gridDim, blockDim, m_function_info,
                    kernel_trace_info->cuda_stream_id) {
  m_parser = parser;
  m_tconfig = config;
  m_kernel_trace_info = kernel_trace_info;
  m_was_launched = false;

  // resolve the binary version
  if (kernel_trace_info->binary_verion == HOPPER_H100_BINART_VERSION)
    OpcodeMap = &Hopper_OpcodeMap;
  else if (kernel_trace_info->binary_verion == AMPERE_RTX_BINART_VERSION ||
           kernel_trace_info->binary_verion == AMPERE_A100_BINART_VERSION)
    OpcodeMap = &Ampere_OpcodeMap;
  else if (kernel_trace_info->binary_verion == VOLTA_BINART_VERSION)
    OpcodeMap = &Volta_OpcodeMap;
  else if (kernel_trace_info->binary_verion == PASCAL_TITANX_BINART_VERSION ||
           kernel_trace_info->binary_verion == PASCAL_P100_BINART_VERSION)
    OpcodeMap = &Pascal_OpcodeMap;
  else if (kernel_trace_info->binary_verion == KEPLER_BINART_VERSION)
    OpcodeMap = &Kepler_OpcodeMap;
  else if (kernel_trace_info->binary_verion == TURING_BINART_VERSION)
    OpcodeMap = &Turing_OpcodeMap;
  else {
    printf("unsupported binary version: %d\n",
           kernel_trace_info->binary_verion);
    fflush(stdout);
    exit(0);
  }
}

void trace_kernel_info_t::get_next_threadblock_traces(
    std::vector<std::vector<inst_trace_t> *> threadblock_traces) {
  m_parser->get_next_threadblock_traces(
      threadblock_traces, m_kernel_trace_info->trace_verion,
      m_kernel_trace_info->enable_lineinfo, m_kernel_trace_info->pipeReader);
}

types_of_operands get_oprnd_type(op_type op, special_ops sp_op) {
  switch (op) {
    case SP_OP:
    case SFU_OP:
    case SPECIALIZED_UNIT_2_OP:
    case SPECIALIZED_UNIT_3_OP:
    case DP_OP:
    case LOAD_OP:
    case STORE_OP:
      return FP_OP;
    case INTP_OP:
    case SPECIALIZED_UNIT_4_OP:
      return INT_OP;
    case ALU_OP:
      if ((sp_op == FP__OP) || (sp_op == TEX__OP) || (sp_op == OTHER_OP))
        return FP_OP;
      else if (sp_op == INT__OP)
        return INT_OP;
    default:
      return UN_OP;
  }
}

bool trace_warp_inst_t::parse_from_trace_struct(
    const inst_trace_t &trace,
    const std::unordered_map<std::string, OpcodeChar> *OpcodeMap,
    const class trace_config *tconfig,
    const class kernel_trace_t *kernel_trace_info) {
  // fill the inst_t and warp_inst_t params
  set_cuda_cta_id(trace.cta_id);
  set_cuda_cluster_cta_id(trace.cluster_cta_id);
  set_cuda_cluster_id(trace.cluster_id);
  set_cuda_cluster_rank(trace.cluster_rank);
  // fill active mask
  active_mask_t active_mask = trace.mask;
  set_active(active_mask);

  // fill and initialize common params
  m_decoded = true;
  pc = (address_type)trace.m_pc;

  isize =
      16;  // starting from MAXWELL isize=16 bytes (including the control bytes)
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
  }
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
  }

  is_vectorin = 0;
  is_vectorout = 0;
  pred = 0;
  ar1 = 0;
  ar2 = 0;
  memory_op = no_memory_op;
  data_size = 0;
  op = ALU_OP;
  sp_op = OTHER_OP;
  mem_op = NOT_TEX;
  const_cache_operand = 0;
  oprnd_type = UN_OP;

  // get the opcode
  std::vector<std::string> opcode_tokens = trace.get_opcode_tokens();
  std::string opcode1 = opcode_tokens[0];

  std::unordered_map<std::string, OpcodeChar>::const_iterator it =
      OpcodeMap->find(opcode1);
  if (it != OpcodeMap->end()) {
    m_opcode = it->second.opcode;
    op = (op_type)(it->second.opcode_category);
    const std::unordered_map<unsigned, unsigned> *OpcPowerMap = &OpcodePowerMap;
    std::unordered_map<unsigned, unsigned>::const_iterator it2 =
        OpcPowerMap->find(m_opcode);
    if (it2 != OpcPowerMap->end()) sp_op = (special_ops)(it2->second);
    oprnd_type = get_oprnd_type(op, sp_op);
  } else {
    std::cout << "ERROR:  undefined instruction : " << trace.opcode
              << " Opcode: " << opcode1 << std::endl;
    assert(0 && "undefined instruction");
  }
  std::string opcode = trace.opcode;
  if (opcode1 == "MUFU") {  // Differentiate between different MUFU operations
                            // for power model
    if ((opcode == "MUFU.SIN") || (opcode == "MUFU.COS")) sp_op = FP_SIN_OP;
    if ((opcode == "MUFU.EX2") || (opcode == "MUFU.RCP")) sp_op = FP_EXP_OP;
    if (opcode == "MUFU.RSQ") sp_op = FP_SQRT_OP;
    if (opcode == "MUFU.LG2") sp_op = FP_LG_OP;
  }

  if (opcode1 == "IMAD") {  // Differentiate between different IMAD operations
                            // for power model
    if ((opcode == "IMAD.MOV") || (opcode == "IMAD.IADD")) sp_op = INT__OP;
  }

  // Fence instructions
  if (opcode == "FENCE.VIEW.ASYNC.S") {
    // TODO microbenchmark for cluster fence
    set_proxy_fence(true);
    set_fence_proxy_kind(ASYNC_SHARED_CTA);
  }

  // Handling SYNCS instructions
  if (opcode1 == "SYNCS") {
    // All SYNCS instructions supported below need register value tracing
    if (trace.reg_src_vals.empty()) {
      printf(
          "Error: SYNCS instruction %s require register value tracing at PC: "
          "0x%llx, exiting "
          "execution\n",
          opcode.c_str(), (address_type)trace.m_pc);
      exit(1);
    }

    syncs_operand operand;
    // Get mbarrier addresses from the trace address info
    for (int i = 0; i < WARP_SIZE; i++) {
      operand.addr[i] = trace.memadd_info->addrs[i];
    }
    if (opcode == "SYNCS.EXCH.64") {  // mbarrier.init
      set_syncs_op(SYNCS_INIT);
      // SYNCS.EXCH.64 format:
      // SYNCS.EXCH.64 URA, [URB], URC
      // URA is destination register
      // URB is a memory reference operand in NVBit
      // URC is the count register, so it is the second source register
      // but based on Hopper SASS code dump, there are some bit manipulation
      // prior to this instructions, so we need to undo this to get the actual
      // thread count
      std::array<uint32_t, WARP_SIZE> thread_counts = trace.reg_src_vals[1];
      for (int i = 0; i < WARP_SIZE; i++) {
        // First right shift by 1
        thread_counts[i] >>= 1;
        // Then substract by 0x100000
        thread_counts[i] -= 0x100000;
        // Finally, take the negation
        thread_counts[i] = -thread_counts[i];
      }
      memcpy(operand.u.init.count, thread_counts.data(),
             sizeof(operand.u.init.count));
    } else if (opcode == "SYNCS.ARRIVE.TRANS64") {  // mbarrier.arrive.expect_tx
      set_syncs_op(SYNCS_ARRIVE_EXPECT_TX);
      // SYNCS.ARRIVE.TRANS64 format:
      // SYNCS.ARRIVE.TRANS64 RA, [RB+URC], RD
      // RA: destination register
      // RB, URC: memory reference operand in NVBit
      // RD: expected byte count
      for (int i = 0; i < WARP_SIZE; i++) {
        // This instruction increase arrival count by 1
        operand.u.arrive.count[i] = 1;
      }
      memcpy(operand.u.arrive.txCount, trace.reg_src_vals[1].data(),
             sizeof(operand.u.arrive.txCount));
    } else if (opcode.find("SYNCS.ARRIVE") !=
               std::string::npos) {  // mbarrier.arrive
      set_syncs_op(SYNCS_ARRIVE);
      // Initialize the arrival count and transaction count to 0
      memset(operand.u.arrive.count, 0, sizeof(operand.u.arrive.count));
      memset(operand.u.arrive.txCount, 0, sizeof(operand.u.arrive.txCount));
      // Handle other variants
      if (opcode.find("ART0") != std::string::npos) {
        // Arrival count is the register value in RD above
        memcpy(operand.u.arrive.count, trace.reg_src_vals[1].data(),
               sizeof(operand.u.arrive.count));
      } else if (opcode.find("A1T0") != std::string::npos) {
        // Arrival 1, transaction 0
        for (int i = 0; i < WARP_SIZE; i++) {
          // This instruction increase arrival count by 1
          operand.u.arrive.count[i] = 1;
        }
      } else if (opcode.find("A0TR") != std::string::npos) {
        // Arrival 0, transaction count based on register value in RD
        memcpy(operand.u.arrive.txCount, trace.reg_src_vals[1].data(),
               sizeof(operand.u.arrive.txCount));
      } else if (opcode.find("A0TX") != std::string::npos) {
        // Arrival 0, complete transaction count based on register value in RD
        set_syncs_op(SYNCS_COMPELTE_TX);
        memcpy(operand.u.complete_tx.txCount, trace.reg_src_vals[1].data(),
               sizeof(operand.u.complete_tx.txCount));
      } else if (opcode.find("A0T1") != std::string::npos) {
        // Arrival 0, transaction count 1
        for (int i = 0; i < WARP_SIZE; i++) {
          operand.u.arrive.txCount[i] = 1;
        }
      } else {
        printf(
            "Error: Unsupported SYNCS ARRIVE variant: %s, aborting execution\n",
            opcode.c_str());
        exit(1);
      }
    } else if (opcode == "SYNCS.PHASECHK.TRANS64") {  // mbarrier.test_wait
      set_syncs_op(SYNCS_TEST_WAIT);
    } else if (opcode ==
               "SYNCS.PHASECHK.TRANS64.TRYWAIT") {  // mbarrier.try_wait
      set_syncs_op(SYNCS_TRY_WAIT);
      // SYNCS.PHASECHK.TRANS64.TRYWAIT format:
      // SYNCS.PHASECHK.TRANS64.TRYWAIT PA, [RB+URC], RD
      // PA: predicate register
      // RB, URC: memory reference operand in NVBit, to the mbarrier
      // RD: prior phase count
      memcpy(operand.u.wait.phase, trace.reg_src_vals[1].data(),
             sizeof(operand.u.wait.phase));
    } else {
      printf(
          "Error: Unsupported SYNCS instruction: %s at PC: 0x%llx, exiting "
          "execution\n",
          opcode.c_str(), (address_type)trace.m_pc);
      exit(1);
    }
    set_syncs_operand(operand);
  }

  // fill regs information
  num_regs = trace.reg_srcs_num + trace.reg_dsts_num;
  num_operands = num_regs;
  outcount = trace.reg_dsts_num;
  // For now, we only model regular registers usage
  auto convert_reg_num = [&](const trace_reg_t &reg) -> uint32_t {
    if (reg.type == REG)
      return reg.num + 1;  // Increment by one because GPGPU-sim starts
                           // from R1, while SASS starts from R0
    else
      return 0;
  };
  for (unsigned m = 0; m < trace.reg_dsts_num; ++m) {
    out[m] = convert_reg_num(trace.reg_dest[m]);
    arch_reg.dst[m] = convert_reg_num(trace.reg_dest[m]);
  }

  incount = trace.reg_srcs_num;
  for (unsigned m = 0; m < trace.reg_srcs_num; ++m) {
    in[m] = convert_reg_num(trace.reg_src[m]);
    arch_reg.src[m] = convert_reg_num(trace.reg_src[m]);
  }

  // fill latency and initl
  tconfig->set_latency(op, latency, initiation_interval);

  // fill addresses
  if (trace.tma_memadd_info != nullptr) {
    data_size = trace.tma_memadd_info->width;
    std::bitset<WARP_SIZE> exec_mask(trace.mask);
    // TMA issues on uniform pipeline, so it will issue once even with full mask
    if (exec_mask.count() != 0) {
      set_tma_access_addrs(trace.tma_memadd_info->addrs);
      // Set TMA mbar address and byte count
      set_tma_mbar_addr(trace.tma_mbar_addr);
      set_tma_byte_count(trace.tma_byte_count);
      set_tma_oob_byte_count(trace.tma_oob_byte_count);
      set_tma_multicast(trace.tma_is_multicast);
      set_tma_multicast_cta_mask(trace.tma_multicast_cta_mask);
    } else {
      // If the TMA instruction is predicated off, we set the addr array to
      // to empty, so it won't generate any memory access during
      // generate_mem_access()
      set_tma_access_addrs(std::vector<uint64_t>());
      set_tma_mbar_addr(0);
      set_tma_byte_count(0);
      set_tma_oob_byte_count(0);
      set_tma_multicast(false);
      set_tma_multicast_cta_mask(0);
    }
  } else if (trace.memadd_info != NULL) {
    data_size = trace.memadd_info->width;
    for (unsigned i = 0; i < warp_size(); ++i)
      set_addr(i, trace.memadd_info->addrs[i]);
  }

  // handle special cases and fill memory space
  switch (m_opcode) {
    case OP_LDC:  // handle Load from Constant
      data_size = 4;
      memory_op = memory_load;
      const_cache_operand = 1;
      space.set_type(const_space);
      cache_op = CACHE_ALL;
      break;
    case OP_LDG:
    // LDGSTS is loading the values needed directly from the global memory to
    // shared memory. Before this feature, the values need to be loaded to
    // registers first, then store to the shared memory.
    case OP_LDGSTS:  // Add for memcpy_async
    case OP_LDL:
      assert(data_size > 0);
      memory_op = memory_load;
      cache_op = CACHE_ALL;
      if (m_opcode == OP_LDL)
        space.set_type(local_space);
      else
        space.set_type(global_space);
      // Add for LDGSTS instruction
      if (m_opcode == OP_LDGSTS) m_is_ldgsts = true;
      // check the cache scope, if its strong GPU, then bypass L1
      if ((trace.check_opcode_contain(opcode_tokens, "STRONG") &&
           trace.check_opcode_contain(opcode_tokens, "GPU")) ||
          trace.check_opcode_contain(opcode_tokens, "BYPASS")) {
        cache_op = CACHE_GLOBAL;
      }
      break;
    case OP_STG:
    case OP_STL:
      assert(data_size > 0);
      memory_op = memory_store;
      cache_op = CACHE_ALL;
      if (m_opcode == OP_STL)
        space.set_type(local_space);
      else
        space.set_type(global_space);
      break;
    case OP_ATOMG:
    case OP_RED:
    case OP_ATOM:
      assert(data_size > 0);
      memory_op = memory_load;
      op = LOAD_OP;
      space.set_type(global_space);
      m_isatomic = true;
      cache_op = CACHE_GLOBAL;  // all the atomics should be done at L2
      break;
    case OP_LDS:
      assert(data_size > 0);
      memory_op = memory_load;
      space.set_type(shared_space);
      break;
    case OP_STS:
      assert(data_size > 0);
      memory_op = memory_store;
      space.set_type(shared_space);
      break;
    case OP_ATOMS:
      assert(data_size > 0);
      m_isatomic = true;
      memory_op = memory_load;
      space.set_type(shared_space);
      break;
    case OP_LDSM:
      assert(data_size > 0);
      space.set_type(shared_space);
      break;
    case OP_ST:
    case OP_LD:
      assert(data_size > 0);
      if (m_opcode == OP_LD)
        memory_op = memory_load;
      else
        memory_op = memory_store;
      // resolve generic loads
      if (kernel_trace_info->shmem_base_addr == 0 ||
          kernel_trace_info->local_base_addr == 0) {
        // shmem and local addresses are not set
        // assume all the mem reqs are shared by default
        space.set_type(shared_space);
      } else {
        // check the first active address
        for (unsigned i = 0; i < warp_size(); ++i)
          if (active_mask.test(i)) {
            if (trace.memadd_info->addrs[i] >=
                    kernel_trace_info->shmem_base_addr &&
                trace.memadd_info->addrs[i] <
                    kernel_trace_info->local_base_addr)
              space.set_type(shared_space);
            else if (trace.memadd_info->addrs[i] >=
                         kernel_trace_info->local_base_addr &&
                     trace.memadd_info->addrs[i] <
                         kernel_trace_info->local_base_addr +
                             LOCAL_MEM_SIZE_MAX) {
              space.set_type(local_space);
              cache_op = CACHE_ALL;
            } else {
              space.set_type(global_space);
              cache_op = CACHE_ALL;
            }
            break;
          }
      }

      break;
    case OP_BAR:
      if (opcode.find("ARV") != std::string::npos) {
        bar_type = ARRIVE;
      } else {
        bar_type = SYNC;
      }
      bar_id = static_cast<unsigned>(trace.imm);
      if (trace.imm2 != 0) {
        bar_count = static_cast<unsigned>(trace.imm2);
      } else {
        bar_count = static_cast<unsigned>(-1);
      }
      break;
    // LDGDEPBAR is to form a group containing the previous LDGSTS instructions
    // that have not been grouped yet. In the implementation, a group number
    // will be assigned once the instruction is met.
    case OP_LDGDEPBAR:
      m_is_ldgdepbar = true;
      break;
    // UMACMDFLUSH is to used to form a bulk group containing all the previous
    // stores due to TMA instructions. Here we reuse the same logic as LDGSTS
    // groups.
    case OP_UTMACMDFLUSH:
      m_is_tma_cmdflush = true;
      break;
    // DEPBAR is served as a warp-wise barrier that is only effective for LDGSTS
    // instructions. It is associated with a immediate value. The immediate
    // value indicates the last N LDGDEPBAR groups to not wait once the
    // instruction is met. For example, if the immediate value is 1, then the
    // last group is able to proceed even with DEPBAR present; if the immediate
    // value is 0, then all of the groups need to finish before proceed.
    case OP_DEPBAR:
      m_is_depbar = true;
      m_depbar_group_no = trace.imm;
      break;
    case OP_HADD2:
    case OP_HADD2_32I:
    case OP_HFMA2:
    case OP_HFMA2_32I:
    case OP_HMUL2_32I:
    case OP_HSET2:
    case OP_HSETP2:
      initiation_interval =
          initiation_interval / 2;  // FP16 has 2X throughput than FP32
      if (initiation_interval <
          1)  // Make sure initiaion interval never goes below 1
        initiation_interval = 1;
      break;
    // TMA instructions
    case OP_UBLKCP:
      // Determine if this is a load or store by checking the source and
      // destination space Src: shared, dst: global -> store Src: global, dst:
      // shared -> load Src: shared, dst: shared -> nop
      if (opcode.find("S.G") != std::string::npos) {
        memory_op = memory_load;
        space.set_type(global_space);
      } else if (opcode.find("G.S") != std::string::npos) {
        memory_op = memory_store;
        space.set_type(global_space);
      } else {
        memory_op = no_memory_op;
      }
      // TMA access L2 only
      cache_op = CACHE_GLOBAL;
      break;
    case OP_UBLKRED:
      memory_op = memory_store;
      // TMA access L2 only
      cache_op = CACHE_GLOBAL;
      space.set_type(global_space);
      break;
    case OP_UBLKPF:
    case OP_UTMAPF:
      // TODO Handle prefetch properly, for now, assume it is a NOP
      printf("Weili: WARNING: TMA prefetch not implemented\n");
      break;
    case OP_UTMALDG:
      memory_op = memory_load;
      // TMA access L2 only
      cache_op = CACHE_GLOBAL;
      space.set_type(global_space);
      break;
    case OP_UTMASTG:
    case OP_UTMAREDG:
      memory_op = memory_store;
      // TMA access L2 only
      cache_op = CACHE_GLOBAL;
      space.set_type(global_space);
      break;
    case OP_SYNCS:
      // Although SYNCS will access shared memory, it will need be handled
      // differently then a normal load, so we set it to no_memory_op here to
      // avoid confusion with is_load() method.
      memory_op = no_memory_op;
      // SYNCS will access shared memory, set the space here
      // to handle the ldst_unit's writeback correctly
      space.set_type(shared_space);
      break;
    case OP_ARRIVES:
      // Handle ARRIVES.LDGSTSBAR instructions, which update the mbarrier
      // when all prior LDGSTS instructions are done
      if (opcode.find("ARRIVES.LDGSTSBAR.64.TRANSCNT") != std::string::npos ||
          opcode.find("ARRIVES.LDGSTSBAR.64.ARVCNT") != std::string::npos) {
        m_is_ldgsts_arrives_mbar = true;
        if (opcode.find("ARRIVES.LDGSTSBAR.64.ARVCNT") != std::string::npos) {
          // ARVCNT: arrive-on operation, decrement pending_thread_count
          // Corresponds to cp.async.mbarrier.arrive.noinc PTX instruction
          m_is_ldgsts_arrives_arvcnt = true;
        }
        for (int i = 0; i < WARP_SIZE; i++) {
          m_ldgsts_arrives_mbar_addr[i] = trace.memadd_info->addrs[i];
        }
      } else {
        printf("WARNING: Unsupported ARRIVES instruction: %s, ignoring it\n",
               opcode.c_str());
      }
      break;
    case OP_QGMMA:
    case OP_HGMMA:
    case OP_IGMMA: {
      // For GMMA instructions, their latency and initiation interval depends
      // solely on N size MxNxK is the MMA tile shape
      std::string mxnxk = opcode_tokens[1];

      // Extract M, N, K values from the string (format: "MxNxK")
      std::stringstream ss(mxnxk);
      std::string m_str, n_str, k_str;

      if (std::getline(ss, m_str, 'x') && std::getline(ss, n_str, 'x') &&
          std::getline(ss, k_str)) {
        int M = std::stoi(m_str);
        int N = std::stoi(n_str);
        int K = std::stoi(k_str);

        auto iter = Hopper_GMMA_N_Latency_Initiation_Interval_Mapping.find(N);
        if (iter != Hopper_GMMA_N_Latency_Initiation_Interval_Mapping.end()) {
          // Set the instruction latency based on N size
          latency = iter->second.first;
          initiation_interval = iter->second.second;
        } else {
          std::string err_msg = "Unsupported N size: " + std::to_string(N);
          assert(false && err_msg.c_str());
        }
      } else {
        std::string err_msg =
            "Failed to extract M, N, K values from the string: " + mxnxk;
        assert(false && err_msg.c_str());
      }

      // Set m_is_gmma_commit_group here
      m_is_gmma_commit_group = trace.is_gmma_commit_group;
    } break;
    case OP_WARPGROUP:
      // Here we handle Hopper's WARPGROUP instructions
      // For now, we just convert WARPGROUP.DEPBAR into a DEPBAR instruction
      // waiting for GMMA group
      if (opcode.find("WARPGROUP.DEPBAR") != std::string::npos) {
        m_is_depbar = true;
        m_depbar_group_no = trace.imm;
      }
      break;
    default:
      break;
  }

  return true;
}

trace_config::trace_config() {}

void trace_config::reg_options(option_parser_t opp) {
  option_parser_register(opp, "-trace", OPT_CSTR, &g_traces_filename,
                         "traces kernel file"
                         "traces kernel file directory",
                         "./traces/kernelslist.g");

  option_parser_register(opp, "-trace_opcode_latency_initiation_int", OPT_CSTR,
                         &trace_opcode_latency_initiation_int,
                         "Opcode latencies and initiation for integers in "
                         "trace driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_sp", OPT_CSTR,
                         &trace_opcode_latency_initiation_sp,
                         "Opcode latencies and initiation for sp in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_dp", OPT_CSTR,
                         &trace_opcode_latency_initiation_dp,
                         "Opcode latencies and initiation for dp in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_sfu", OPT_CSTR,
                         &trace_opcode_latency_initiation_sfu,
                         "Opcode latencies and initiation for sfu in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_tensor",
                         OPT_CSTR, &trace_opcode_latency_initiation_tensor,
                         "Opcode latencies and initiation for tensor in trace "
                         "driven mode <latency,initiation>",
                         "4,1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-trace_opcode_latency_initiation_spec_op_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &trace_opcode_latency_initiation_specialized_op[j],
                           "specialized unit config"
                           " <latency,initiation>",
                           "4,4");
  }
}

void trace_config::parse_config() {
  sscanf(trace_opcode_latency_initiation_int, "%u,%u", &int_latency, &int_init);
  sscanf(trace_opcode_latency_initiation_sp, "%u,%u", &fp_latency, &fp_init);
  sscanf(trace_opcode_latency_initiation_dp, "%u,%u", &dp_latency, &dp_init);
  sscanf(trace_opcode_latency_initiation_sfu, "%u,%u", &sfu_latency, &sfu_init);
  sscanf(trace_opcode_latency_initiation_tensor, "%u,%u", &tensor_latency,
         &tensor_init);

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    sscanf(trace_opcode_latency_initiation_specialized_op[j], "%u,%u",
           &specialized_unit_latency[j], &specialized_unit_initiation[j]);
  }
}
void trace_config::set_latency(unsigned category, unsigned &latency,
                               unsigned &initiation_interval) const {
  initiation_interval = latency = 1;

  switch (category) {
    case ALU_OP:
    case INTP_OP:
    case BRANCH_OP:
    case CALL_OPS:
    case RET_OPS:
      latency = int_latency;
      initiation_interval = int_init;
      break;
    case SP_OP:
      latency = fp_latency;
      initiation_interval = fp_init;
      break;
    case DP_OP:
      latency = dp_latency;
      initiation_interval = dp_init;
      break;
    case SFU_OP:
      latency = sfu_latency;
      initiation_interval = sfu_init;
      break;
    case TENSOR_CORE_OP:
      latency = tensor_latency;
      initiation_interval = tensor_init;
      break;
    default:
      break;
  }
  // for specialized units
  if (category >= SPEC_UNIT_START_ID) {
    unsigned spec_id = category - SPEC_UNIT_START_ID;
    assert(spec_id >= 0 && spec_id < SPECIALIZED_UNIT_NUM);
    latency = specialized_unit_latency[spec_id];
    initiation_interval = specialized_unit_initiation[spec_id];
  }
}

void trace_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new trace_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                    m_shader_stats, m_memory_stats);
}

void trace_simt_core_cluster::create_shader_core_ctx() {
  m_core = new shader_core_ctx *[m_config->n_simt_cores_per_cluster];
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    m_core[i] = new trace_shader_core_ctx(m_gpu, this, sid, m_cluster_id,
                                          m_config, m_mem_config, m_stats);
    m_core_sim_order.push_back(i);
  }
}

void trace_shader_core_ctx::create_shd_warp() {
  m_warp.resize(m_config->max_warps_per_shader);
  for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
    m_warp[k] = new trace_shd_warp_t(this, m_config->warp_size);
  }
}

void trace_shader_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                    const warp_inst_t *pI,
                                                    unsigned *pc,
                                                    unsigned *rpc) {
  // In trace-driven mode, we assume no control hazard
  assert(pI != NULL && "Unexpexted behaviour , inst should not be null");
  *pc = pI->pc;
  *rpc = pI->pc;
}

const active_mask_t &trace_shader_core_ctx::get_active_mask(
    unsigned warp_id, const warp_inst_t *pI) {
  // For Trace-driven, the active mask already set in traces, so
  // just read it from the inst
  return pI->get_active_mask();
}

unsigned trace_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  if (kernel.no_more_ctas_to_run()) {
    return 0;  // finished!
  }

  if (kernel.more_threads_in_cta()) {
    kernel.increment_thread_id();
  }

  if (!kernel.more_threads_in_cta()) kernel.increment_cta_id();

  return 1;
}

void trace_shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                       unsigned end_thread, unsigned ctaid,
                                       int cta_size, kernel_info_t &kernel) {
  // call base class
  shader_core_ctx::init_warps(cta_id, start_thread, end_thread, ctaid, cta_size,
                              kernel);

  // then init traces
  unsigned start_warp = start_thread / m_config->warp_size;
  unsigned end_warp = end_thread / m_config->warp_size +
                      ((end_thread % m_config->warp_size) ? 1 : 0);

  init_traces(start_warp, end_warp, kernel);
}

const warp_inst_t *trace_shader_core_ctx::get_next_inst(unsigned warp_id,
                                                        address_type pc) {
  // read the inst from the traces
  trace_shd_warp_t *m_trace_warp =
      static_cast<trace_shd_warp_t *>(m_warp[warp_id]);
  const trace_warp_inst_t *ret = m_trace_warp->get_next_trace_inst();
  if (ret == NULL && m_trace_warp->trace_done()) {
    // Block warp from exiting if:
    // 1. There are still instructions in the pipeline
    // 2. The warp is waiting at a barrier
    // 3. The warp still has outstanding stores
    // 4. The warp still has pending writes
    if (!m_warp[warp_id]->inst_in_pipeline() &&
        !m_barriers.warp_waiting_at_barrier(warp_id) &&
        m_warp[warp_id]->stores_done() &&
        !m_scoreboard->pendingWrites(warp_id)) {
      for (unsigned t = 0; t < m_warp_size; t++) {
        if (m_warp[warp_id]->test_active(t)) {
          m_warp[warp_id]->set_completed(t);
        }
      }
      m_barriers.warp_exit(warp_id);
    }
  }
  return ret;
}

void trace_shader_core_ctx::updateSIMTStack(unsigned warpId,
                                            warp_inst_t *inst) {
  // No SIMT-stack in trace-driven  mode
}

void trace_shader_core_ctx::init_traces(unsigned start_warp, unsigned end_warp,
                                        kernel_info_t &kernel) {
  std::vector<std::vector<inst_trace_t> *> threadblock_traces;
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    m_trace_warp->clear();
    threadblock_traces.push_back(&(m_trace_warp->warp_traces));
  }
  trace_kernel_info_t &trace_kernel =
      static_cast<trace_kernel_info_t &>(kernel);
  trace_kernel.get_next_threadblock_traces(threadblock_traces);

  // set the pc from the traces and ignore the functional model
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
    m_trace_warp->set_kernel(&trace_kernel);
  }
}

void trace_shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst,
                                                          unsigned t,
                                                          unsigned tid) {
  if (inst.isatomic()) m_warp[inst.warp_id()]->inc_n_atomic();

  if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
    new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
    unsigned num_addrs;
    num_addrs = translate_local_memaddr(
        inst.get_addr(t), tid,
        m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
        inst.data_size, (new_addr_type *)localaddrs);
    inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
  }
}

void trace_shader_core_ctx::func_exec_inst(warp_inst_t &inst) {
  for (unsigned t = 0; t < m_warp_size; t++) {
    if (inst.active(t)) {
      unsigned warpId = inst.warp_id();
      unsigned tid = m_warp_size * warpId + t;

      // virtual function
      checkExecutionStatusAndUpdate(inst, t, tid);
    }
  }
  // here, we generate memory acessess and set the status if thread (done?)
  if (inst.is_load() || inst.is_store()) {
    inst.generate_mem_accesses();
  }
}

void trace_shader_core_ctx::issue_warp(register_set &warp,
                                       const warp_inst_t *pI,
                                       const active_mask_t &active_mask,
                                       unsigned warp_id, unsigned sch_id) {
  shader_core_ctx::issue_warp(warp, pI, active_mask, warp_id, sch_id);

  // delete warp_inst_t class here, it is not required anymore by gpgpu-sim
  // after issue
  delete pI;
}
