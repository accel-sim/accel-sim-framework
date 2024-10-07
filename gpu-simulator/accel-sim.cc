#include "accel-sim.h"
#include "accelsim_version.h"

accel_sim_framework::accel_sim_framework(std::string config_file,
                                         std::string trace_file) {
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  m_gpgpu_context = new gpgpu_context();

  // mimic argv
  unsigned argc = 5;
  const char *argv[] = {"accel-sim.out", "-config", config_file.c_str(),
                        "-trace", trace_file.c_str()};

  gpgpu_sim *m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  tracer = trace_parser(tconfig.get_traces_filename());

  tconfig.parse_config();

  init();
}

accel_sim_framework::accel_sim_framework(int argc, const char **argv) {
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  m_gpgpu_context = new gpgpu_context();

  m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  tracer = trace_parser(tconfig.get_traces_filename());

  tconfig.parse_config();

  init();
}

void accel_sim_framework::simulation_loop() {
  // for each kernel
  // load file
  // parse and create kernel info
  // launch
  // while loop till the end of the end kernel execution
  // prints stats

  if (finished_graphics == tracer.graphics_count) {
    printf("No graphics kernel parsed\n");
    printf("STEP1 - rendering done at %llu\n", m_gpgpu_sim->gpu_tot_sim_cycle);
    m_gpgpu_sim->all_graphics_done = true;
    graphics_done = true;
  }
  if (finished_computes == tracer.compute_count) {
    printf("No compute kernel parsed\n");
    printf("STEP1 - computes done at %llu\n", m_gpgpu_sim->gpu_tot_sim_cycle);
    m_gpgpu_sim->gpu_compute_end_cycle = m_gpgpu_sim->gpu_tot_sim_cycle;
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

  if (m_gpgpu_sim->get_config().gpgpu_slicer) {
    m_gpgpu_sim->concurrent_granularity =
        m_gpgpu_sim->get_config().num_shader();
    m_gpgpu_sim->dynamic_sm_count = m_gpgpu_sim->get_config().num_shader() / 2;
  }

  while (commandlist_index < commandlist.size() || !kernels_info.empty()) {
    parse_commandlist();

    // Launch all kernels within window that are on a stream that isn't already
    // running
    for (auto k : kernels_info) {
      bool stream_busy = false;
      for (auto s : busy_streams) {
        if (s == k->get_cuda_stream_id()) stream_busy = true;
      }
      if (!stream_busy && m_gpgpu_sim->can_start_kernel() &&
          !k->was_launched()) {
        if ((launched_mesa ==
                 m_gpgpu_sim->get_config().get_max_concurrent_kernel() * 3 /
                     4 &&
             k->is_graphic_kernel)) {
          continue;
        }
        std::cout << "launching kernel name: " << k->get_name()
                  << " uid: " << k->get_uid()
                  << " cuda_stream_id: " << k->get_cuda_stream_id()
                  << std::endl;
        if (!k->is_graphic_kernel) {
          m_gpgpu_sim->gipc = 0;
        } else {
          // graphics
          m_gpgpu_sim->cipc = 0;
          launched_mesa++;
        }
        m_gpgpu_sim->launch(k);
        k->set_launched();
        busy_streams.push_back(k->get_cuda_stream_id());
      }
    }

    unsigned finished_kernel_uid = simulate();
    // cleanup finished kernel
    if (finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit() ||
        !m_gpgpu_sim->active()) {
      cleanup(finished_kernel_uid);
    }

    if (sim_cycles) {
      m_gpgpu_sim->update_stats();
      m_gpgpu_context->print_simulation_time();
    }

    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      printf(
          "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
          "instructions) **\n");
      fflush(stdout);
      break;
    }

    if (finished_graphics == tracer.graphics_count) {
      printf("All graphics kernels finished one iteration\n");
      printf("STEP1 - rendering done at %llu\n",
             m_gpgpu_sim->gpu_tot_sim_cycle);
      m_gpgpu_sim->all_graphics_done = true;
      graphics_done = true;
    }
    if (finished_computes == tracer.compute_count && !computes_done) {
      printf("All compute kernels finished one iteration\n");
      printf("STEP1 - computes done at %llu\n", m_gpgpu_sim->gpu_tot_sim_cycle);
      m_gpgpu_sim->gpu_compute_end_cycle = m_gpgpu_sim->gpu_tot_sim_cycle;
      m_gpgpu_sim->all_compute_done = true;
      computes_done = true;
    }
    if (graphics_done && computes_done) {
      printf(
          "GPGPU-Sim: ** break due to finishing all kernels one iteration "
          "**\n");
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
      m_gpgpu_sim->all_graphics_done = false;

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
      m_gpgpu_sim->all_compute_done = false;
      printf("relaunching compute kernels\n");
    }
  }
}

void accel_sim_framework::parse_commandlist() {
  // gulp up as many commands as possible - either cpu_gpu_mem_copy
  // or kernel_launch - until the vector "kernels_info" has reached
  // the window_size or we have read every command from commandlist
  while (kernels_info.size() < window_size &&
         commandlist_index < commandlist.size()) {
    trace_kernel_info_t *kernel_info = NULL;
    if (commandlist[commandlist_index].m_type ==
        command_type::cpu_gpu_mem_copy) {
      size_t addre, Bcount;
      size_t per_CTA = -1;
      tracer.parse_memcpy_info(commandlist[commandlist_index].command_string,
                               addre, Bcount, per_CTA);
      if (commandlist[commandlist_index].command_string.find("MemcpyVulkan") ==
          std::string::npos) {
        std::cout << "launching memcpy command : "
                  << commandlist[commandlist_index].command_string << std::endl;
        m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount, false);
      } else {
        assert(per_CTA != (unsigned)-1);
        kernel_vb_addr.push_back(addre);
        kernel_vb_size.push_back(Bcount);
        kernel_per_CTA.push_back(per_CTA);
        graphics_commands.push_back(commandlist[commandlist_index]);
      }
      commandlist_index++;
    } else if (commandlist[commandlist_index].m_type ==
               command_type::kernel_launch) {
      // Read trace header info for window_size number of kernels
      kernel_trace_t *kernel_trace_info = tracer.parse_kernel_info(
          commandlist[commandlist_index].command_string);
      kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context,
                                       &tconfig, &tracer);

      if (kernel_info->is_graphic_kernel) {
        graphics_commands.push_back(commandlist[commandlist_index]);
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
        compute_commands.push_back(commandlist[commandlist_index]);
      }

      kernels_info.push_back(kernel_info);
      std::cout << "Header info loaded for kernel command : "
                << commandlist[commandlist_index].command_string << std::endl;
      commandlist_index++;
    } else {
      // unsupported commands will fail the simulation
      assert(0 && "Undefined Command");
    }
  }
}

void accel_sim_framework::cleanup(unsigned finished_kernel) {
  trace_kernel_info_t *k = NULL;
  unsigned long long finished_kernel_cuda_stream_id = -1;
  for (unsigned j = 0; j < kernels_info.size(); j++) {
    k = kernels_info.at(j);
    if (k->get_uid() == finished_kernel ||
        m_gpgpu_sim->cycle_insn_cta_max_hit() || !m_gpgpu_sim->active()) {
      for (unsigned int l = 0; l < busy_streams.size(); l++) {
        if (busy_streams.at(l) == k->get_cuda_stream_id()) {
          finished_kernel_cuda_stream_id = k->get_cuda_stream_id();
          busy_streams.erase(busy_streams.begin() + l);
          break;
        }
      }
      tracer.kernel_finalizer(k->get_trace_info());
      // delete k->entry(); // erased somewhere else
      // delete k;
      if (m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm) {
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
            assert(
                m_gpgpu_sim->getShaderCoreConfig()->n_simt_cores_per_cluster ==
                1);
            m_gpgpu_sim->getSIMTCluster(cluster)->get_core(0)->shader_inst = 0;
          }
          m_gpgpu_sim->dynamic_sm_count =
              m_gpgpu_sim->get_config().dynamic_sm_count;
        } else {
        }
      }

      kernels_info.erase(kernels_info.begin() + j);
      if (!m_gpgpu_sim->cycle_insn_cta_max_hit() && m_gpgpu_sim->active())
        break;
    }
  }
  assert(k);
  m_gpgpu_sim->print_stats(finished_kernel_cuda_stream_id);
}

unsigned accel_sim_framework::simulate() {
  unsigned finished_kernel_uid = 0;
  do {
    if (!m_gpgpu_sim->active()) break;

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
  return finished_kernel_uid;
}

trace_kernel_info_t *accel_sim_framework::create_kernel_info(
    kernel_trace_t *kernel_trace_info, gpgpu_context *m_gpgpu_context,
    trace_config *config, trace_parser *parser) {
  gpgpu_ptx_sim_info info;
  info.smem = kernel_trace_info->shmem;
  info.regs = kernel_trace_info->nregs;
  dim3 gridDim(kernel_trace_info->grid_dim_x, kernel_trace_info->grid_dim_y,
               kernel_trace_info->grid_dim_z);
  dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y,
                kernel_trace_info->tb_dim_z);
  trace_function_info *function_info =
      new trace_function_info(info, m_gpgpu_context);
  function_info->set_name(kernel_trace_info->kernel_name.c_str());
  trace_kernel_info_t *kernel_info = new trace_kernel_info_t(
      gridDim, blockDim, function_info, parser, config, kernel_trace_info);

  return kernel_info;
}

gpgpu_sim *accel_sim_framework::gpgpu_trace_sim_init_perf_model(
    int argc, const char *argv[], gpgpu_context *m_gpgpu_context,
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
      opp);  // register GPU microrachitecture options
  m_config->reg_options(opp);

  option_parser_cmdline(opp, argc, argv);  // parse configuration options
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