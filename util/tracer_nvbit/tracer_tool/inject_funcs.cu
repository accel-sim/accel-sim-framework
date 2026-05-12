/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#include <stdint.h>
#include <stdio.h>

#include <cstdarg>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"

/* get_cluster_ctaid and get_cluster_nctaid are now provided by NVBit v1.8
 * utils.h */

// Helper functions to get this CTA's cluster rank in a cluster
// and its cluster id in the grid
__device__ __forceinline__ int get_cluster_rank(void) {
  int ret;
  asm("mov.u32 %0, %cluster_ctarank;" : "=r"(ret));
  return ret;
}

__device__ __forceinline__ int4 get_clusterid(void) {
  int4 ret;
  asm("mov.u32 %0, %clusterid.x;" : "=r"(ret.x));
  asm("mov.u32 %0, %clusterid.y;" : "=r"(ret.y));
  asm("mov.u32 %0, %clusterid.z;" : "=r"(ret.z));
  return ret;
}

/* Instrumentation function that we want to inject, please note the use of
 *  extern "C" __device__ __noinline__
 *    To prevent "dead"-code elimination by the compiler.
 */
extern "C" __device__ __noinline__ void
instrument_inst(int pred, int opcode_id, int32_t vpc, bool is_tma,
                uint64_t tma_param_handle, uint32_t tma_param_handle_size,
                bool is_mem, uint64_t addr, int32_t width, int32_t desReg,
                int32_t srcReg1, int32_t srcReg2, int32_t srcReg3,
                int32_t srcReg4, int32_t srcReg5, int32_t srcNum,
                uint64_t immediate, uint64_t immediate2, uint64_t pchannel_dev,
                uint64_t ptotal_dynamic_instr_counter,
                uint64_t preported_dynamic_instr_counter, uint64_t pstop_report,
                uint32_t line_num, uint32_t instr_idx, uint32_t desRegVal,
                uint32_t srcReg1Val, uint32_t srcReg2Val, uint32_t srcReg3Val,
                uint32_t srcReg4Val, uint32_t srcReg5Val, uint32_t auxRegVal,
                bool is_gmma_commit_group) {
  const int active_mask = __ballot_sync(__activemask(), 1);
  const int predicate_mask = __ballot_sync(__activemask(), pred);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  if ((*((bool *)pstop_report))) {
    if (first_laneid == laneid) {
      atomicAdd((unsigned long long *)ptotal_dynamic_instr_counter, 1);
      return;
    }
  }

  inst_trace_t trace;

  // Set the trace header
  int4 cta = get_ctaid();
#if __CUDA_ARCH__ >= 900
  int4 cluster_cta = get_cluster_ctaid();
  int4 cluster = get_clusterid();
  int cluster_rank = get_cluster_rank();
#else
  int4 cluster_cta = {0, 0, 0, 0}; // Dummy values for pre-SM90
  int4 cluster = {0, 0, 0, 0};     // Dummy values for pre-SM90
  int cluster_rank = 0;            // Dummy value for pre-SM90
#endif
  int uniqe_threadId = threadIdx.z * blockDim.y * blockDim.x +
                       threadIdx.y * blockDim.x + threadIdx.x;
  trace.active_mask = active_mask;
  trace.predicate_mask = predicate_mask;
  trace.cta_id_x = cta.x;
  trace.cta_id_y = cta.y;
  trace.cta_id_z = cta.z;
  trace.cluster_cta_id_x = cluster_cta.x;
  trace.cluster_cta_id_y = cluster_cta.y;
  trace.cluster_cta_id_z = cluster_cta.z;
  trace.cluster_id_x = cluster.x;
  trace.cluster_id_y = cluster.y;
  trace.cluster_id_z = cluster.z;
  trace.cluster_rank = cluster_rank;
  trace.warpid_tb = uniqe_threadId / 32;
  trace.warpid_sm = get_warpid();
  trace.sm_id = get_smid();
  trace.opcode_id = opcode_id;
  trace.vpc = vpc;
  trace.line_num = line_num;
  trace.instr_idx = instr_idx;
  trace.is_gmma_commit_group = is_gmma_commit_group;

  if (!is_tma) {
    // For regular instructions
    trace.inst_type = TracerInstrType::INST_REGULAR;
    if (is_mem) {
      /* collect memory address information */
      for (int i = 0; i < 32; i++) {
        trace.inst.regular.addrs[i] = __shfl_sync(active_mask, addr, i);
      }
      trace.inst.regular.width = width;
      trace.inst.regular.is_mem = true;
    } else {
      trace.inst.regular.is_mem = false;
    }
    trace.inst.regular.GPRDst = desReg;
    trace.inst.regular.GPRSrcs[0] = srcReg1;
    trace.inst.regular.GPRSrcs[1] = srcReg2;
    trace.inst.regular.GPRSrcs[2] = srcReg3;
    trace.inst.regular.GPRSrcs[3] = srcReg4;
    trace.inst.regular.GPRSrcs[4] = srcReg5;
    trace.inst.regular.numSrcs = srcNum;
    trace.inst.regular.imm = immediate;
    trace.inst.regular.imm2 = immediate2;

    for (int tid = 0; tid < 32; tid++) {
      trace.inst.regular.desRegVal[tid] =
          __shfl_sync(active_mask, desRegVal, tid);
      trace.inst.regular.srcRegVals[0][tid] =
          __shfl_sync(active_mask, srcReg1Val, tid);
      trace.inst.regular.srcRegVals[1][tid] =
          __shfl_sync(active_mask, srcReg2Val, tid);
      trace.inst.regular.srcRegVals[2][tid] =
          __shfl_sync(active_mask, srcReg3Val, tid);
      trace.inst.regular.srcRegVals[3][tid] =
          __shfl_sync(active_mask, srcReg4Val, tid);
      trace.inst.regular.srcRegVals[4][tid] =
          __shfl_sync(active_mask, srcReg5Val, tid);
      trace.inst.regular.auxRegVals[tid] =
          __shfl_sync(active_mask, auxRegVal, tid);
    }
  } else {
    // For TMA instructions
    trace.inst_type = TracerInstrType::INST_TMA;
    memset(trace.inst.tma.tma_param_handle, 0,
           sizeof(trace.inst.tma.tma_param_handle));
    memcpy(trace.inst.tma.tma_param_handle, (uint8_t *)tma_param_handle,
           tma_param_handle_size);
    trace.inst.tma.tma_param_handle_ptr = tma_param_handle;
    trace.inst.tma.tma_param_handle_size = tma_param_handle_size;
  }

  // First active lane pushes information on the channel
  if (first_laneid == laneid) {
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&trace, sizeof(inst_trace_t));
    atomicAdd((unsigned long long *)ptotal_dynamic_instr_counter, 1);
    atomicAdd((unsigned long long *)preported_dynamic_instr_counter, 1);
  }
}
