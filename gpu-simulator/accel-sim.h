#include <math.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../ISA_Def/trace_opcode.h"
#include "../trace-parser/trace_parser.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpu_context.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "trace_driven.h"

class accel_sim_framework {
 public:
  accel_sim_framework(int argc, const char **argv);
  accel_sim_framework(std::string config_file, std::string trace_file);

  void init() {
    active = false;
    sim_cycles = false;
    window_size = 0;
    commandlist_index = 0;

    assert(m_gpgpu_context);
    assert(m_gpgpu_sim);

    concurrent_kernel_sm =
        m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
    window_size = concurrent_kernel_sm
                      ? m_gpgpu_sim->get_config().get_max_concurrent_kernel()
                      : 1;
    assert(window_size > 0);
    commandlist = tracer.parse_commandlist_file();

    kernels_info.reserve(window_size);
  }
  void simulation_loop();
  void parse_commandlist();
  void cleanup(unsigned finished_kernel);
  unsigned simulate();
  trace_kernel_info_t *create_kernel_info(kernel_trace_t *kernel_trace_info,
                                          gpgpu_context *m_gpgpu_context,
                                          trace_config *config,
                                          trace_parser *parser);
  gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                  gpgpu_context *m_gpgpu_context,
                                  trace_config *m_config);


 private:
  gpgpu_context *m_gpgpu_context;
  trace_config tconfig;
  trace_parser tracer;
  gpgpu_sim *m_gpgpu_sim;

  bool concurrent_kernel_sm;
  bool active;
  bool sim_cycles;
  unsigned window_size;
  unsigned commandlist_index;

  std::vector<unsigned long long> busy_streams;
  std::vector<trace_kernel_info_t *> kernels_info;
  std::vector<trace_command> commandlist;

};