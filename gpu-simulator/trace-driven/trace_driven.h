// Copyright (c) 2018-2021, Mahmoud Khairy, Vijay Kandiah, Timothy Rogers, Tor
// M. Aamodt, Nikos Hardavellas
// Northwestern University, Purdue University, The
// University of British Columbia
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef TRACE_DRIVEN_H
#define TRACE_DRIVEN_H

#include "../ISA_Def/trace_opcode.h"
#include "../trace-parser/trace_parser.h"
#include "abstract_hardware_model.h"
#include "gpgpu-sim/shader.h"

class trace_function_info : public function_info {
 public:
  trace_function_info(const struct gpgpu_ptx_sim_info &info,
                      gpgpu_context *m_gpgpu_context)
      : function_info(0, m_gpgpu_context) {
    m_kernel_info = info;
  }

  virtual const struct gpgpu_ptx_sim_info *get_kernel_info() const {
    return &m_kernel_info;
  }

  virtual const void set_kernel_info(const struct gpgpu_ptx_sim_info &info) {
    m_kernel_info = info;
  }

  virtual ~trace_function_info() {}
};

class trace_warp_inst_t : public warp_inst_t {
 public:
  trace_warp_inst_t() {
    m_opcode = 0;
    should_do_atomic = false;
  }

  trace_warp_inst_t(const class core_config *config) : warp_inst_t(config) {
    m_opcode = 0;
    should_do_atomic = false;
  }

  bool parse_from_trace_struct(
      const inst_trace_t &trace,
      const std::unordered_map<std::string, OpcodeChar> *OpcodeMap,
      const class trace_config *tconfig,
      const class kernel_trace_t *kernel_trace_info);

 private:
  unsigned m_opcode;
};

class trace_kernel_info_t : public kernel_info_t {
 public:
  trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                      trace_function_info *m_function_info,
                      trace_parser *parser, class trace_config *config,
                      kernel_trace_t *kernel_trace_info);

  void get_next_threadblock_traces(
      std::vector<std::vector<inst_trace_t> *> threadblock_traces);

  unsigned long long get_cuda_stream_id() {
    return m_kernel_trace_info->cuda_stream_id;
  }

  kernel_trace_t *get_trace_info() { return m_kernel_trace_info; }

  bool was_launched() { return m_was_launched; }

  void set_launched() { m_was_launched = true; }

 private:
  trace_config *m_tconfig;
  const std::unordered_map<std::string, OpcodeChar> *OpcodeMap;
  trace_parser *m_parser;
  kernel_trace_t *m_kernel_trace_info;
  bool m_was_launched;

  friend class trace_shd_warp_t;
};

class trace_config {
 public:
  trace_config();

  void set_latency(unsigned category, unsigned &latency,
                   unsigned &initiation_interval) const;
  void parse_config();
  void reg_options(option_parser_t opp);
  char *get_traces_filename() { return g_traces_filename; }

 private:
  unsigned int_latency, fp_latency, dp_latency, sfu_latency, tensor_latency;
  unsigned int_init, fp_init, dp_init, sfu_init, tensor_init;
  unsigned specialized_unit_latency[SPECIALIZED_UNIT_NUM];
  unsigned specialized_unit_initiation[SPECIALIZED_UNIT_NUM];

  char *g_traces_filename;
  char *trace_opcode_latency_initiation_int;
  char *trace_opcode_latency_initiation_sp;
  char *trace_opcode_latency_initiation_dp;
  char *trace_opcode_latency_initiation_sfu;
  char *trace_opcode_latency_initiation_tensor;
  char *trace_opcode_latency_initiation_specialized_op[SPECIALIZED_UNIT_NUM];
};

class trace_shd_warp_t : public shd_warp_t {
 public:
  trace_shd_warp_t(class shader_core_ctx *shader, unsigned warp_size)
      : shd_warp_t(shader, warp_size) {
    trace_pc = 0;
    m_kernel_info = NULL;
  }

  std::vector<inst_trace_t> warp_traces;
  const trace_warp_inst_t *get_next_trace_inst();
  void clear();
  bool trace_done();
  address_type get_start_trace_pc();
  virtual address_type get_pc();
  virtual kernel_info_t *get_kernel_info() const { return m_kernel_info; }
  void set_kernel(trace_kernel_info_t *kernel_info) {
    m_kernel_info = kernel_info;
  }

 private:
  unsigned trace_pc;
  trace_kernel_info_t *m_kernel_info;
};

class trace_gpgpu_sim : public gpgpu_sim {
 public:
  trace_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {
    createSIMTCluster();
  }

  virtual void createSIMTCluster();
};

class trace_simt_core_cluster : public simt_core_cluster {
 public:
  trace_simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                          const shader_core_config *config,
                          const memory_config *mem_config,
                          class shader_core_stats *stats,
                          class memory_stats_t *mstats)
      : simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats) {
    create_shader_core_ctx();
  }

  virtual void create_shader_core_ctx();
};

class trace_shader_core_ctx : public shader_core_ctx {
 public:
  trace_shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                        unsigned shader_id, unsigned tpc_id,
                        const shader_core_config *config,
                        const memory_config *mem_config,
                        shader_core_stats *stats)
      : shader_core_ctx(gpu, cluster, shader_id, tpc_id, config, mem_config,
                        stats) {
    create_front_pipeline();
    create_shd_warp();
    create_schedulers();
    create_exec_pipeline();
  }

  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid);
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          kernel_info_t &kernel);
  virtual void func_exec_inst(warp_inst_t &inst);
  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu);
  virtual void create_shd_warp();
  virtual const warp_inst_t *get_next_inst(unsigned warp_id, address_type pc);
  virtual void updateSIMTStack(unsigned warpId, warp_inst_t *inst);
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc);
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI);
  virtual void issue_warp(register_set &warp, const warp_inst_t *pI,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned sch_id);

 private:
  void init_traces(unsigned start_warp, unsigned end_warp,
                   kernel_info_t &kernel);
};

types_of_operands get_oprnd_type(op_type op, special_ops sp_op);

#endif
