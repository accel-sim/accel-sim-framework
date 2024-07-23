// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

#include <execinfo.h>
#include <signal.h>

#include "gpgpu_context.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "../ISA_Def/trace_opcode.h"
#include "trace_driven.h"
#include "../trace-parser/trace_parser.h"
#include "accelsim_version.h"

/* TO DO:
 * NOTE: the current version of trace-driven is functionally working fine,
 * but we still need to improve traces compression and simulation speed.
 * This includes:
 *
 * 1- Prefetch concurrent thread that prefetches traces from disk (to not be
 * limited by disk speed)
 *
 * 2- traces compression format a) cfg format and remove
 * thread/block Id from the head and b) using zlib library to save in binary format
 *
 * 3- Efficient memory improvement (save string not objects - parse only 10 in
 * the buffer)
 *
 * 4- Seeking capability - thread scheduler (save tb index and warp
 * index info in the traces header)
 *
 * 5- Get rid off traces intermediate files -
 * change the tracer
 */

void handler(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           class trace_config *m_config);

trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser);


int main(int argc, const char **argv) {
  signal(SIGSEGV, handler);   // install our handler
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  gpgpu_context *m_gpgpu_context = new gpgpu_context();
  trace_config tconfig;

  gpgpu_sim *m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  trace_parser tracer(tconfig.get_traces_filename());

  tconfig.parse_config();

  // for each kernel
  // load file
  // parse and create kernel info
  // launch
  // while loop till the end of the end kernel execution
  // prints stats
  bool concurrent_kernel_sm =  m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
  unsigned window_size = concurrent_kernel_sm ? 1024 : 1;
  assert(window_size > 0);
  std::vector<trace_command> commandlist = tracer.parse_commandlist_file();
  std::vector<trace_command> compute_commands;
  std::vector<trace_command> graphics_commands;
  std::vector<unsigned long> busy_streams;
  std::vector<trace_kernel_info_t*> kernels_info;
  kernels_info.reserve(window_size);
  printf("%u MESA kernels parsed\n", tracer.graphics_count);
  /*
  // hardcode the previous frame info for now
  unsigned *compute_cycles = new unsigned[tracer.compute_count]{
    // harris conrner 6
    40312, 21679, 21514, 84511, 15334, 5813,
    // klt tracker 7
    27505,26807,30498,35058,10376,11590,95894,
    // distortion 4
    447050,62830,135116,22671,
    // opt flow 20
    40302,8366,6423,6243,6230,5822,5420,10917,5814,5414,10268,5417,5414,10237,5417,5414,10251,5417,5414,10233
  };
  m_gpgpu_sim->predicted_render_cycle = 0;
  for (unsigned i = 0; i < tracer.compute_count; i++) {
    m_gpgpu_sim->compute_cycles[i + 1 + tracer.graphics_count] = compute_cycles[i];
  }
  int64_t predicted = 0;
  */
  /*
  // below: render_passes_dev
  unsigned *graphics_cycles =
      new unsigned[tracer.graphics_count]{
          36952, 10779, 12322, 8366,   15129, 8667,  72379, 42623,
          12394, 10192, 9447,  265291, 18146, 83690, 12627, 35149,
          34125, 85216, 24885, 16419,  9376,  13124, 7731,  48662,
          65976, 9916,  23612, 8121,   7713,  9941,  44402, 8027,
          35996, 10778, 8316,  7270,   28490, 85504, 36277, 115834,
          27931, 18698, 29922, 9702,   28770, 15377, 21360, 12481};
  */
  /*
  // below: render_passes_2k
  unsigned *graphics_cycles = new unsigned[tracer.graphics_count]{
      36617, 23539,  12037, 10327,   14963, 10888,  72830, 152444,
      12415, 19372,  9269,  1080844, 18204, 343158, 12613, 121515,
      34131, 348325, 24511, 42931,   9205,  33789,  7734,  180056,
      67422, 18844,  23449, 8920,    7911,  19150,  44392, 8804,
      35907, 24019,  8355,  8363,    28418, 341819, 36797, 480426,
      27655, 57341,  29215, 18006,   28875, 41267,  21608, 28977};
  double *graphics_error =
      new double[tracer.graphics_count]{
          -0.009106505, 0.00196099,   -0.001769781, -0.008679811, -0.004621328,
          0.000987261,  0.017523652,  0.003755501,  0.014833661,  0.005469204,
          -0.227559841, -0.0015228,   0.022571026,  -0.005920848, -0.061801381,
          -0.022776911, -0.004001749, -0.00891049,  -0.077729955, 0.03054575,
          0.000377782,  0.002648109,  -0.000641255, -0.000588002
      };
  for (unsigned i = 0; i < tracer.graphics_count-1; i+=2) {
    m_gpgpu_sim->grpahics_error[i + 1] = graphics_error[i/2];
    m_gpgpu_sim->grpahics_error[i + 2] = graphics_error[i/2];

    unsigned cycle = (double)graphics_cycles[i] * (1.0 - graphics_error[i / 2]);
    m_gpgpu_sim->predicted_kernel_cycles[i] = cycle;
    predicted += cycle;

    cycle = (double)graphics_cycles[i + 1] * (1.0 - graphics_error[i / 2]);
    m_gpgpu_sim->predicted_kernel_cycles[i + 1] = cycle;
    predicted += cycle;
  }
  m_gpgpu_sim->predicted_render_cycle = predicted;
  m_gpgpu_sim->gpu_last_frame_cycle = 4047657;
  // m_gpgpu_sim->gpu_last_frame_cycle = 2969324;
  m_gpgpu_sim->gpu_last_compute_cycle = 1400930;
  for (unsigned i = 0; i < tracer.graphics_count; i++) {
    m_gpgpu_sim->last_frame_kernels_elapsed_time[i + 1] = graphics_cycles[i];
  }
  for (unsigned i = 0; i < tracer.compute_count; i++) {
    m_gpgpu_sim
        ->last_frame_kernels_elapsed_time[i + 1 + tracer.graphics_count] =
        compute_cycles[i];
  }
  */
  unsigned i = 0;
  unsigned last_launched_vertex = -1;
  unsigned last_grpahics_stream_id = -1;
  unsigned launched_mesa = 0;
  std::vector<unsigned long> kernel_vb_addr;
  std::vector<unsigned long> kernel_vb_size;
  std::vector<unsigned long> kernel_per_CTA;
  unsigned finished_computes = 0;
  unsigned finished_graphics = 0;
  bool computes_done = false;
  bool graphics_done = false;
  m_gpgpu_sim->start_compute = true;
  unsigned long graphics_stream_id = 0xDEADBEEF; 
  if (finished_graphics == tracer.graphics_count) {
      printf("No graphics kernel parsed\n");
      printf("STEP1 - rendering done at %llu\n", m_gpgpu_sim->gpu_tot_sim_cycle);
      m_gpgpu_sim->graphics_done = true;
      m_gpgpu_sim->all_graphics_done = true;
      graphics_done = true;
  }
  if (finished_computes == tracer.compute_count) {
      printf("No compute kernel parsed\n");
      printf("STEP1 - computes done at %llu\n", m_gpgpu_sim->gpu_tot_sim_cycle);
      m_gpgpu_sim->gpu_compute_end_cycle = m_gpgpu_sim->gpu_tot_sim_cycle;
      m_gpgpu_sim->compute_done = true;
      m_gpgpu_sim->all_compute_done = true;
      computes_done = true;
    }

    if (m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm) {
      if (m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_finegrain) {
        m_gpgpu_sim->concurrent_mode = m_gpgpu_sim->FINEGRAIN;
        m_gpgpu_sim->concurrent_granularity = 6;
        m_gpgpu_sim->dynamic_sm_count =
            m_gpgpu_sim->get_config().dynamic_sm_count;
        printf("defualt dynamic ratio %d\n", m_gpgpu_sim->dynamic_sm_count);
      } else {
        m_gpgpu_sim->concurrent_mode = m_gpgpu_sim->MPS;
        m_gpgpu_sim->concurrent_granularity =
            m_gpgpu_sim->get_config().num_shader();
        m_gpgpu_sim->dynamic_sm_count = m_gpgpu_sim->get_config().mps_sm_count;
      }
    }

    if(m_gpgpu_sim->get_config().gpgpu_slicer) {
      m_gpgpu_sim->concurrent_granularity = m_gpgpu_sim->get_config().num_shader();
      m_gpgpu_sim->dynamic_sm_count = m_gpgpu_sim->get_config().num_shader() / 2;
    }
  while (i < commandlist.size() || !kernels_info.empty()) {
    //gulp up as many commands as possible - either cpu_gpu_mem_copy 
    //or kernel_launch - until the vector "kernels_info" has reached
    //the window_size or we have read every command from commandlist
    while (kernels_info.size() < window_size && i < commandlist.size()) {
      trace_kernel_info_t *kernel_info = NULL;
      if (commandlist[i].m_type == command_type::cpu_gpu_mem_copy) {
        size_t addre, Bcount;
        size_t per_CTA = -1;
        tracer.parse_memcpy_info(commandlist[i].command_string, addre, Bcount, per_CTA);
        if (commandlist[i].command_string.find("MemcpyVulkan") ==
            std::string::npos) {
          // std::cout << "launching memcpy command : "
          //           << commandlist[i].command_string << std::endl;
          m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount, false);
        } else {
          assert(per_CTA != -1);
          // std::cout << "Saving MemcpyVulkan for CTA launch : "
          //           << commandlist[i].command_string << std::endl;
          kernel_vb_addr.push_back(addre);
          kernel_vb_size.push_back(Bcount);
          kernel_per_CTA.push_back(per_CTA);
          graphics_commands.push_back(commandlist[i]);
        }
        i++;
      } else if (commandlist[i].m_type == command_type::kernel_launch) {
        // Read trace header info for window_size number of kernels
        kernel_trace_t* kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string);
        kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
        kernel_info->prerequisite_kernel = -1;
        if (kernel_info->is_graphic_kernel) {
          graphics_commands.push_back(commandlist[i]);
          unsigned kernel_id = kernel_info->get_uid();
          if (kernel_info->get_name().find("VERTEX") != std::string::npos) {
            // is vertex shader
            last_launched_vertex = kernel_id;
            kernel_trace_info->cuda_stream_id = graphics_stream_id;
            last_grpahics_stream_id = graphics_stream_id;
            graphics_stream_id++;
          } else {
            assert(kernel_info->get_name().find("FRAG") != std::string::npos);
            kernel_trace_info->cuda_stream_id = last_grpahics_stream_id;
          }

          // for (auto vb = kernel_vb_size.begin(); vb != kernel_vb_size.end();
          //      vb++) {
          //   unsigned byte_per_CTA = *vb / kernel_info->num_blocks();
          //   if (byte_per_CTA != 0) {
          //     *vb = byte_per_CTA;
          //   }
          // }
          // save kernel info
          m_gpgpu_sim->vb_addr[kernel_id] = kernel_vb_addr;
          m_gpgpu_sim->vb_size[kernel_id] = kernel_vb_size;
          m_gpgpu_sim->vb_size_per_cta[kernel_id] = kernel_per_CTA;
          // clear buffers for next kernel
          kernel_vb_addr.clear();
          kernel_vb_size.clear();
          kernel_per_CTA.clear();
        } else {
          assert(kernel_trace_info->cuda_stream_id < 0xDEADBEEF ||
                 kernel_trace_info->cuda_stream_id > 0XDEAFBEEF + 1024);
          kernel_info->prerequisite_kernel = -1;
          compute_commands.push_back(commandlist[i]);
        }
        kernels_info.push_back(kernel_info);
        m_gpgpu_sim->update_stats_size(kernel_info->get_uid());
        std::cout << "Header info loaded for kernel command : " << commandlist[i].command_string << std::endl;
        i++;
      }
      else{
        //unsupported commands will fail the simulation
        assert(0 && "Undefined Command");
      }
    }
    // Launch all kernels within window that are on a stream that isn't already running
    for (auto k : kernels_info) {
      bool stream_busy = false;
      for (auto s: busy_streams) {
        if (s == k->get_cuda_stream_id())
          stream_busy = true;
      }
      if (!stream_busy && m_gpgpu_sim->can_start_kernel() && !k->was_launched()) {
        if ((launched_mesa ==
                 m_gpgpu_sim->get_config().get_max_concurrent_kernel() * 3 /
                     4 &&
             k->is_graphic_kernel)) {
          // if ((launched_mesa == 63 && k->is_graphic_kernel)) {
          continue;
        }
        std::cout << "launching kernel name: " << k->get_name() << " uid: " << k->get_uid() << std::endl;
        std::string kernel_name = k->get_name();
        if (!k->is_graphic_kernel) {
          m_gpgpu_sim->compute_done = false;
          m_gpgpu_sim->gipc = 0;
        } else {
          // graphics
          m_gpgpu_sim->cipc = 0;
          m_gpgpu_sim->graphics_done = false;
          launched_mesa++;
        }
        
        m_gpgpu_sim->launch(k);
        k->set_launched();
        busy_streams.push_back(k->get_cuda_stream_id());
      }
    }

    bool active = false;
    bool sim_cycles = false;
    unsigned finished_kernel_uid = 0;

    do {
      if (!m_gpgpu_sim->active())
        break;

      // performance simulation
      if (m_gpgpu_sim->active()) {
        m_gpgpu_sim->cycle();
        sim_cycles = true;
        m_gpgpu_sim->deadlock_check();
      } else {
        if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
          m_gpgpu_context->the_gpgpusim->g_stream_manager
              ->stop_all_running_kernels();
          break;
        }
      }

      active = m_gpgpu_sim->active();
      finished_kernel_uid = m_gpgpu_sim->finished_kernel();
    } while (active && !finished_kernel_uid);

    // cleanup finished kernel
    if (finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit()
        || !m_gpgpu_sim->active()) {
      trace_kernel_info_t* k = NULL;
      for (unsigned j = 0; j < kernels_info.size(); j++) {
        k = kernels_info.at(j);
        if (k->get_uid() == finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit()
            || !m_gpgpu_sim->active()) {
          for (int l = 0; l < busy_streams.size(); l++) {
            if (busy_streams.at(l) == k->get_cuda_stream_id()) {
              busy_streams.erase(busy_streams.begin()+l);
              break;
            }
          }
          tracer.kernel_finalizer(k->get_trace_info());
          // delete k->entry();
          // delete k;
          if (m_gpgpu_sim->getShaderCoreConfig()
                    ->gpgpu_concurrent_kernel_sm) {
              if (m_gpgpu_sim->concurrent_mode == m_gpgpu_sim->FINEGRAIN) {
                m_gpgpu_sim->dynamic_sm_count =
                    m_gpgpu_sim->get_config().dynamic_sm_count;
              } else {
                m_gpgpu_sim->dynamic_sm_count =
                    m_gpgpu_sim->get_config().mps_sm_count;
              }
            }
          if (k->is_graphic_kernel) {
            finished_graphics++;
            launched_mesa--;
          } else {
            finished_computes++;

            if (m_gpgpu_sim->get_config().gpgpu_slicer) {
            m_gpgpu_sim->slicer_sampled = false;
            for (unsigned cluster = 0;
                 cluster < m_gpgpu_sim->getShaderCoreConfig()->n_simt_clusters;
                 cluster++) {
              assert(m_gpgpu_sim->getShaderCoreConfig()
                         ->n_simt_cores_per_cluster == 1);
              m_gpgpu_sim->getSIMTCluster(cluster)->get_core(0)->shader_inst =
                  0;
            }
            m_gpgpu_sim->dynamic_sm_count =
                    m_gpgpu_sim->get_config().dynamic_sm_count;
          } else {
          }
          }
          kernels_info.erase(kernels_info.begin()+j);
          if (!m_gpgpu_sim->cycle_insn_cta_max_hit())
            break;
        }
      }
      assert(k);
      assert(finished_kernel_uid);
      m_gpgpu_sim->print_stats(finished_kernel_uid);
    }

    if (sim_cycles) {
      m_gpgpu_sim->update_stats();
      m_gpgpu_context->print_simulation_time();
    }

    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      printf("GPGPU-Sim: ** break due to reaching the maximum cycles (or "
             "instructions) **\n");
      fflush(stdout);
      break;
    }
    /*
    I think not needed anymore. But keep it for now
    // if (finished_graphics == tracer.graphics_count && tracer.graphics_count > 0) {
    //   printf("GPGPU-Sim: ** break due to finishing all graphics kernels **\n");
    //   fflush(stdout);
    //   break;
    // }
    // if (finished_computes == tracer.compute_count && tracer.compute_count > 0) {
    //   printf("GPGPU-Sim: ** break due to finishing all compute kernels **\n");
    //   fflush(stdout);
    //   break;
    // }
    */

    // if (m_gpgpu_sim->all_compute_done && tracer.compute_count > 0) {
    //   // break after the *next kernel after all compute done* is done
    //   break;
    // }

    if (finished_graphics == tracer.graphics_count) {
      printf("All graphics kernels finished one iteration\n");
      printf("STEP1 - rendering done at %llu\n", m_gpgpu_sim->gpu_tot_sim_cycle);
      m_gpgpu_sim->graphics_done = true;
      m_gpgpu_sim->all_graphics_done = true;
      graphics_done = true;
    }
    if (finished_computes == tracer.compute_count && !computes_done) {
      printf("All compute kernels finished one iteration\n");
      printf("STEP1 - computes done at %llu\n", m_gpgpu_sim->gpu_tot_sim_cycle);
      m_gpgpu_sim->gpu_compute_end_cycle = m_gpgpu_sim->gpu_tot_sim_cycle;
      m_gpgpu_sim->compute_done = true;
      m_gpgpu_sim->all_compute_done = true;
      computes_done = true;
    }
    if (graphics_done && computes_done) {
      printf("GPGPU-Sim: ** break due to finishing all kernels one iteration **\n");
      break;
    }

    if (finished_graphics == tracer.graphics_count &&
        tracer.graphics_count > 0 && tracer.compute_count > 0 && 
        m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm &&
        !computes_done) {
      for (auto cmd : graphics_commands) {
        commandlist.push_back(cmd);
      }
      finished_graphics = 0;
      graphics_commands.clear();
      m_gpgpu_sim->graphics_done = false;
      m_gpgpu_sim->all_graphics_done = false;

    //   m_gpgpu_sim->new_frame();

      printf("relaunching graphics kernels\n");
    }
    if (finished_computes == tracer.compute_count &&
        tracer.graphics_count > 0 && tracer.compute_count > 0 &&
        m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm &&
        !graphics_done) {
      for (auto cmd : compute_commands) {
        commandlist.push_back(cmd);
      }
      finished_computes = 0;
      compute_commands.clear();
      m_gpgpu_sim->compute_done = false;
      m_gpgpu_sim->all_compute_done = false;
      printf("relaunching compute kernels\n");
    }
  }
  unsigned long long compute_cycle = m_gpgpu_sim->gpu_compute_end_cycle - m_gpgpu_sim->gpu_compute_start_cycle;
  float compute_slowdown =
      (float)compute_cycle / m_gpgpu_sim->gpu_last_compute_cycle;
  float graphics_slowdown = (float)(m_gpgpu_sim->gpu_tot_sim_cycle -
                                    m_gpgpu_sim->gpu_render_start_cycle) /
                            m_gpgpu_sim->gpu_last_frame_cycle;
  printf(
      "STEP1 - compute start time : %llu, compute end time : %llu, compute "
      "slowdown : %f\n",
      m_gpgpu_sim->gpu_compute_start_cycle,
      m_gpgpu_sim->gpu_compute_end_cycle, compute_slowdown);
  printf(
      "STEP1 - graphics start time : %llu, graphics end time : %llu, rendering "
      "slowdown: %f\n",
      m_gpgpu_sim->gpu_render_start_cycle, m_gpgpu_sim->gpu_tot_sim_cycle,
      m_gpgpu_sim->gpu_tot_sim_cycle - m_gpgpu_sim->gpu_render_start_cycle,
      graphics_slowdown);
  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
}


trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser){

  gpgpu_ptx_sim_info info;
  info.smem = kernel_trace_info->shmem;
  info.regs = kernel_trace_info->nregs;
  dim3 gridDim(kernel_trace_info->grid_dim_x, kernel_trace_info->grid_dim_y, kernel_trace_info->grid_dim_z);
  dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y, kernel_trace_info->tb_dim_z);
  trace_function_info *function_info =
      new trace_function_info(info, m_gpgpu_context);
  function_info->set_name(kernel_trace_info->kernel_name.c_str());
  trace_kernel_info_t *kernel_info =
      new trace_kernel_info_t(gridDim, blockDim, function_info,
    		  parser, config, kernel_trace_info);
  if (kernel_trace_info->kernel_name.find("VERTEX") != std::string::npos ||
      kernel_trace_info->kernel_name.find("FRAG") != std::string::npos) {
    kernel_info->is_graphic_kernel = true;
  }

  return kernel_info;
}

gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           trace_config *m_config) {
  srand(1);
  print_splash();

  option_parser_t opp = option_parser_create();

  m_gpgpu_context->ptx_reg_options(opp);
  m_gpgpu_context->func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);

  m_gpgpu_context->the_gpgpusim->g_the_gpu_config =
      new gpgpu_sim_config(m_gpgpu_context);
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->reg_options(
      opp); // register GPU microrachitecture options
  m_config->reg_options(opp);

  option_parser_cmdline(opp, argc, argv); // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();

  m_gpgpu_context->the_gpgpusim->g_the_gpu = new trace_gpgpu_sim(
      *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config), m_gpgpu_context);

  m_gpgpu_context->the_gpgpusim->g_stream_manager =
      new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
                         m_gpgpu_context->func_sim->g_cuda_launch_blocking);

  m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  return m_gpgpu_context->the_gpgpusim->g_the_gpu;
}
