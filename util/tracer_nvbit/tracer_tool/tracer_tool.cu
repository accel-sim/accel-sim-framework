/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"

#define TRACER_VERSION "5"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int enable_compress = 1;
int print_core_id = 0;
int exclude_pred_off = 1;
int active_from_start = 1;
int lineinfo = 0;
/* used to select region of interest when active from start is 0 */
bool active_region = true;

/* Should we terminate the program once we are done tracing? */
int terminate_after_limit_number_of_kernels_reached = 0;
int user_defined_folders = 0;

/* Use xz to compress the *.trace file */
int xz_compress_trace = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

std::string user_folder = getcwd(NULL, 0);
std::string cwd = getcwd(NULL, 0);
std::string traces_location = cwd + "/traces/";
std::string kernelslist_location = cwd + "/traces/kernelslist";
std::string stats_location = cwd + "/traces/stats.csv";

std::unordered_map<CUcontext, std::string> ctx_kernelslist;
std::unordered_map<CUcontext, std::string> ctx_stats_location;
std::unordered_map<CUcontext, int> ctx_kernelid;
std::unordered_map<CUcontext, FILE*> ctx_resultsFile;

/* kernel instruction counter, updated by the GPU */
uint64_t dynamic_kernel_limit_start =
    0;                                 // 0 means start from the begging kernel
uint64_t dynamic_kernel_limit_end = 0; // 0 means no limit

enum address_format { list_all = 0, base_stride = 1, base_delta = 2 };

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 1,
              "Exclude predicated off instruction from count");
  GET_VAR_INT(lineinfo, "TRACE_LINEINFO", 0,
              "Include source code line info at the start of each traced line. "
              "The target binary must be compiled with -lineinfo or "
              "--generate-line-info");
  GET_VAR_INT(dynamic_kernel_limit_end, "DYNAMIC_KERNEL_LIMIT_END", 0,
              "Limit of the number kernel to be printed, 0 means no limit");
  GET_VAR_INT(dynamic_kernel_limit_start, "DYNAMIC_KERNEL_LIMIT_START", 0,
              "start to report kernel from this kernel id, 0 means starts from "
              "the beginning, i.e. first kernel");
  GET_VAR_INT(
      active_from_start, "ACTIVE_FROM_START", 1,
      "Start instruction tracing from start or wait for cuProfilerStart "
      "and cuProfilerStop. If set to 0, DYNAMIC_KERNEL_LIMIT options have no "
      "effect");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(enable_compress, "TOOL_COMPRESS", 1, "Enable traces compression");
  GET_VAR_INT(print_core_id, "TOOL_TRACE_CORE", 0,
              "write the core id in the traces");
  GET_VAR_INT(
      terminate_after_limit_number_of_kernels_reached, "TERMINATE_UPON_LIMIT",
      0, "Stop the process once the current kernel > DYNAMIC_KERNEL_LIMIT_END");
  GET_VAR_INT(user_defined_folders, "USER_DEFINED_FOLDERS", 0,
              "Uses the user defined "
              "folder TRACES_FOLDER path environment");
  GET_VAR_INT(xz_compress_trace, "TRACE_FILE_COMPRESS", 1,
              "Create xz-compressed trace"
              "file");
  std::string pad(100, '-');
  printf("%s\n", pad.c_str());

  if (active_from_start == 0) {
    active_region = false;
  }
  char * usr_defined_folder = std::getenv("TRACES_FOLDER");
  if (usr_defined_folder != NULL)
    user_folder = usr_defined_folder;

}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* instrument each memory instruction adding a call to the above instrumentation
 * function */
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions) {
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    if (!already_instrumented.insert(f).second) {
      continue;
    }

    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose) {
      printf("Inspecting function %s at address 0x%lx\n",
             nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx,f), true);
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
      uint32_t line_num = 0;
      // Temporary workaround for a bug in NVBit 1.7.4, which does not correctly handle `call.rel`.  
      // Instrumenting this instruction leads to illegal memory access.  
      // Refer to: https://github.com/NVlabs/NVBit/issues/142#issue-2911561744  
      if(!strcmp(instr->getOpcode(), "CALL.REL.NOINC")){
        printf("Warning: Ignoring CALL.REL.NOINC (NVBit 1.7.4 bug)\n");
        continue;
      } 

      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }

      if (verbose) {
        instr->printDecoded();
      }

      if (lineinfo) {
        char *file_name, *dir_name;
        nvbit_get_line_info(ctx, func, instr->getOffset(), &file_name,
                            &dir_name, &line_num);
      }

      if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = instr->getOpcode();
      }

      int opcode_id = opcode_to_id_map[instr->getOpcode()];

      /* check all operands. For now, we ignore constant, TEX, predicates and
       * unified registers. We only report vector regisers */
      int src_oprd[MAX_SRC];
      int srcNum = 0;
      int dst_oprd = -1;
      int mem_oper_idx = -1;
      int num_mref = 0;
      uint64_t imm_value = 0;

      for (int i = 0; i < instr->getNumOperands(); ++i) {
        const InstrType::operand_t *op = instr->getOperand(i);
        if (op->type == InstrType::OperandType::MREF) {
          assert(srcNum < MAX_SRC);
          src_oprd[srcNum] = instr->getOperand(i)->u.mref.ra_num;
          srcNum++;
          mem_oper_idx++;
          num_mref++;
          // if(mem_oper_idx == 0){
          //   mem_oper_idx = 1; // loop control
          // }
        } else if (op->type == InstrType::OperandType::REG) {
          if (i == 0) {
            // find dst reg
            dst_oprd = instr->getOperand(0)->u.reg.num;
          } else {
            // find src regs
            assert(srcNum < MAX_SRC);
            src_oprd[srcNum] = instr->getOperand(i)->u.reg.num;
            srcNum++;
          }
        }
        // Add immediate value for DEPBAR instruction
        else if (op->type == InstrType::OperandType::IMM_UINT64) {
          imm_value = instr->getOperand(i)->u.imm_uint64.value;
        }
      }

      do {
        /* insert call to the instrumentation function with its
         * arguments */
        nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);

        /* pass predicate value */
        nvbit_add_call_arg_guard_pred_val(instr);

        /* send opcode and pc */
        nvbit_add_call_arg_const_val32(instr, opcode_id);
        nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());

        /* mem addresses info */
        if (mem_oper_idx >= 0) {
          nvbit_add_call_arg_const_val32(instr, 1);
          assert(num_mref <= 2);
          if (num_mref == 2) { // LDGSTS
            nvbit_add_call_arg_mref_addr64(instr, 1 - mem_oper_idx);
          } else {
            nvbit_add_call_arg_mref_addr64(instr, mem_oper_idx);
          }
          nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());
        } else {
          nvbit_add_call_arg_const_val32(instr, 0);
          nvbit_add_call_arg_const_val64(instr, -1);
          nvbit_add_call_arg_const_val32(instr, -1);
        }

        /* reg info */
        nvbit_add_call_arg_const_val32(instr, dst_oprd);
        for (int i = 0; i < srcNum; i++) {
          nvbit_add_call_arg_const_val32(instr, src_oprd[i]);
        }
        for (int i = srcNum; i < MAX_SRC; i++) {
          nvbit_add_call_arg_const_val32(instr, -1);
        }
        nvbit_add_call_arg_const_val32(instr, srcNum);

        /* immediate info */
        nvbit_add_call_arg_const_val64(instr, imm_value);

        /* add pointer to channel_dev and other counters*/
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
        nvbit_add_call_arg_const_val64(instr,
                                       (uint64_t)&total_dynamic_instr_counter);
        nvbit_add_call_arg_const_val64(
            instr, (uint64_t)&reported_dynamic_instr_counter);
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&stop_report);
        /* Add Source code line number for current instr */
        nvbit_add_call_arg_const_val32(instr, (int)line_num);

        mem_oper_idx--;
      } while (mem_oper_idx >= 0);

      cnt++;
    }
  }
}

__global__ void flush_channel() {
  /* push memory access with negative cta id to communicate the kernel is
   * completed */
  inst_trace_t ma;
  ma.cta_id_x = -1;
  channel_dev.push(&ma, sizeof(inst_trace_t));

  /* flush channel */
  channel_dev.flush();
}

static FILE *resultsFile = NULL;
static FILE *kernelsFile = NULL;
static FILE *statsFile = NULL;
static int kernelid = 1;
static bool first_call = true;

unsigned old_total_insts = 0;
unsigned old_total_reported_insts = 0;

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  if (skip_flag)
    return;

  if (first_call == true) {
    first_call = false;
    std::string traces_folder = user_folder + "/traces";
    if (mkdir(traces_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
      if (errno == EEXIST) {
        // alredy exists
      } else {
        // something else
        std::cout << "cannot create folder error:" << strerror(errno)
                  << std::endl;
        return;
      }
    }

    if (active_from_start && !dynamic_kernel_limit_start ||
        dynamic_kernel_limit_start == 1)
      active_region = true;
    else {
      if (active_from_start)
        active_region = false;
    }

    kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "w");
    statsFile = fopen(ctx_stats_location[ctx].c_str(), "w");
    fprintf(statsFile,
            "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, "
            "#blocks, block_dimX, block_dimY, block_dimZ, #threads, "
            "total_insts, total_reported_insts\n");
    fclose(statsFile);
  }

  if (cbid == API_CUDA_cuMemcpyHtoD_v2) {
    if (!is_exit) {
      cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;
      char buffer[1024];
      kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "a");
      sprintf(buffer, "MemcpyHtoD,0x%016lx,%lld", p->dstDevice, p->ByteCount);
      fprintf(kernelsFile, buffer);
      fprintf(kernelsFile, "\n");
      fclose(kernelsFile);
    }

  } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
             cbid == API_CUDA_cuLaunchKernel) {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

    if (!is_exit) {
      if (active_from_start && dynamic_kernel_limit_start &&
          ctx_kernelid[ctx] == dynamic_kernel_limit_start)
        active_region = true;

      if (terminate_after_limit_number_of_kernels_reached &&
          dynamic_kernel_limit_end != 0 &&
          ctx_kernelid[ctx] > dynamic_kernel_limit_end) {
        exit(0);
      }

      int nregs;
      CUDA_SAFECALL(
          cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

      int shmem_static_nbytes;
      CUDA_SAFECALL(cuFuncGetAttribute(
          &shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

      int binary_version;
      CUDA_SAFECALL(cuFuncGetAttribute(&binary_version,
                                       CU_FUNC_ATTRIBUTE_BINARY_VERSION, p->f));

      instrument_function_if_needed(ctx, p->f);

      if (active_region) {
        nvbit_enable_instrumented(ctx, p->f, true);
        stop_report = false;
      } else {
        nvbit_enable_instrumented(ctx, p->f, false);
        stop_report = true;
      }

      char buffer[2048];
      std::string trace_filename = user_folder + "/traces/"; 
      sprintf(buffer, "%s/kernel-%d-ctx_0x%lx.trace", trace_filename.c_str(),
              ctx_kernelid[ctx], ctx);

      if (!stop_report) {
        if (!xz_compress_trace) {
          ctx_resultsFile[ctx] = fopen(buffer, "w");
          printf("Writing results to %s\n", buffer);
        } else {
          char cmd_buffer[1039];
          sprintf(cmd_buffer, "xz -1 -T0 > %s.xz", buffer);
          ctx_resultsFile[ctx] = popen(cmd_buffer, "w");
          printf("Writing results to %s.xz\n", buffer);
        }

        fprintf(ctx_resultsFile[ctx], "-kernel name = %s\n",
                nvbit_get_func_name(ctx, p->f, true));
        fprintf(ctx_resultsFile[ctx], "-kernel id = %d\n", ctx_kernelid[ctx]);
        fprintf(ctx_resultsFile[ctx], "-grid dim = (%d,%d,%d)\n", p->gridDimX,
                p->gridDimY, p->gridDimZ);
        fprintf(ctx_resultsFile[ctx], "-block dim = (%d,%d,%d)\n", p->blockDimX,
                p->blockDimY, p->blockDimZ);
        fprintf(ctx_resultsFile[ctx], "-shmem = %d\n",
                shmem_static_nbytes + p->sharedMemBytes);
        fprintf(ctx_resultsFile[ctx], "-nregs = %d\n", nregs);
        fprintf(ctx_resultsFile[ctx], "-binary version = %d\n", binary_version);
        fprintf(ctx_resultsFile[ctx], "-cuda stream id = %lu\n", (uint64_t)p->hStream);
        fprintf(ctx_resultsFile[ctx], "-shmem base_addr = 0x%016lx\n",
                (uint64_t)nvbit_get_shmem_base_addr(ctx));
        fprintf(ctx_resultsFile[ctx], "-local mem base_addr = 0x%016lx\n",
                (uint64_t)nvbit_get_local_mem_base_addr(ctx));
        fprintf(ctx_resultsFile[ctx], "-nvbit version = %s\n", NVBIT_VERSION);
        fprintf(ctx_resultsFile[ctx], "-accelsim tracer version = %s\n", TRACER_VERSION);
        fprintf(ctx_resultsFile[ctx], "-enable lineinfo = %d\n", lineinfo);
        fprintf(ctx_resultsFile[ctx], "\n");

        fprintf(ctx_resultsFile[ctx],
                "#traces format = [line_num] PC mask dest_num [reg_dests] "
                "opcode src_num "
                "[reg_srcs] mem_width [adrrescompress?] [mem_addresses] "
                "immediate\n");
        fprintf(ctx_resultsFile[ctx], "\n");
      }

      kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "a");
      // This will be a relative path to the traces file
      
      sprintf(buffer, "kernel-%d-ctx_0x%lx.trace%s", ctx_kernelid[ctx], ctx,
              xz_compress_trace ? ".xz" : "");
      if (!stop_report) {
        fprintf(kernelsFile, buffer);
        fprintf(kernelsFile, "\n");
      }
      fclose(kernelsFile);

      statsFile = fopen(ctx_stats_location[ctx].c_str(), "a");
      unsigned blocks = p->gridDimX * p->gridDimY * p->gridDimZ;
      unsigned threads = p->blockDimX * p->blockDimY * p->blockDimZ;

      fprintf(statsFile, "%s, %s, %d, %d, %d, %d, %d, %d, %d, %d, ", buffer,
              nvbit_get_func_name(ctx, p->f, true), p->gridDimX, p->gridDimY,
              p->gridDimZ, blocks, p->blockDimX, p->blockDimY, p->blockDimZ,
              threads);

      fclose(statsFile);

      ctx_kernelid[ctx]++;
      recv_thread_receiving = true;

    } else {
      /* make sure current kernel is completed */
      cudaDeviceSynchronize();
      assert(cudaGetLastError() == cudaSuccess);

      /* make sure we prevent re-entry on the nvbit_callback when issuing
       * the flush_channel kernel */
      skip_flag = true;

      /* issue flush of channel so we are sure all the memory accesses
       * have been pushed */
      flush_channel<<<1, 1>>>();
      cudaDeviceSynchronize();
      assert(cudaGetLastError() == cudaSuccess);

      /* unset the skip flag */
      skip_flag = false;

      /* wait here until the receiving thread has not finished with the
       * current kernel */
      while (recv_thread_receiving) {
        pthread_yield();
      }

      unsigned total_insts_per_kernel =
          total_dynamic_instr_counter - old_total_insts;
      old_total_insts = total_dynamic_instr_counter;

      unsigned reported_insts_per_kernel =
          reported_dynamic_instr_counter - old_total_reported_insts;
      old_total_reported_insts = reported_dynamic_instr_counter;

      statsFile = fopen(ctx_stats_location[ctx].c_str(), "a");
      fprintf(statsFile, "%d,%d", total_insts_per_kernel,
              reported_insts_per_kernel);
      fprintf(statsFile, "\n");
      fclose(statsFile);

      if (!stop_report) {
        if (!xz_compress_trace) {
          fclose(ctx_resultsFile[ctx]);
        } else {
          pclose(ctx_resultsFile[ctx]);
        }
      }

      if (active_from_start && dynamic_kernel_limit_end &&
          ctx_kernelid[ctx] > dynamic_kernel_limit_end)
        active_region = false;
    }
  } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
    if (!active_from_start) {
      active_region = true;
    }
  } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
    if (!active_from_start) {
      active_region = false;
    }
  }
}

bool is_number(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it))
    ++it;
  return !s.empty() && it == s.end();
}

unsigned get_datawidth_from_opcode(const std::vector<std::string> &opcode) {
  for (unsigned i = 0; i < opcode.size(); ++i) {
    if (is_number(opcode[i])) {
      return (std::stoi(opcode[i], NULL) / 8);
    } else if (opcode[i][0] == 'U' && is_number(opcode[i].substr(1))) {
      // handle the U* case
      unsigned bits;
      sscanf(opcode[i].c_str(), "U%u", &bits);
      return bits / 8;
    }
  }

  return 4; // default is 4 bytes
}

bool check_opcode_contain(const std::vector<std::string> &opcode,
                          std::string param) {
  for (unsigned i = 0; i < opcode.size(); ++i)
    if (opcode[i] == param)
      return true;

  return false;
}

bool base_stride_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                          uint64_t &base_addr, int &stride) {
  // calulcate the difference between addresses
  // write cosnsctive addresses with constant stride in a more
  // compressed way (i.e. start adress and stride)
  bool const_stride = true;
  bool first_bit1_found = false;
  bool last_bit1_found = false;

  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      first_bit1_found = true;
      base_addr = addrs[s];
      if (s < 31 && mask.test(s + 1))
        stride = addrs[s + 1] - addrs[s];
      else {
        const_stride = false;
        break;
      }
    } else if (first_bit1_found && !last_bit1_found) {
      if (mask.test(s)) {
        if (stride != addrs[s] - addrs[s - 1]) {
          const_stride = false;
          break;
        }
      } else
        last_bit1_found = true;
    } else if (last_bit1_found) {
      if (mask.test(s)) {
        const_stride = false;
        break;
      }
    }
  }

  return const_stride;
}

void base_delta_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                         uint64_t &base_addr, std::vector<long long> &deltas) {
  // save the delta from the previous address
  bool first_bit1_found = false;
  uint64_t last_address = 0;
  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      base_addr = addrs[s];
      first_bit1_found = true;
      last_address = addrs[s];
    } else if (mask.test(s) && first_bit1_found) {
      deltas.push_back(addrs[s] - last_address);
      last_address = addrs[s];
    }
  }
}

void *recv_thread_fun(void *args) {
  CUcontext ctx = (CUcontext)args;
  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);
  while (recv_thread_started) {
    uint32_t num_recv_bytes = 0;
    if (recv_thread_receiving &&
        (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        inst_trace_t *ma = (inst_trace_t *)&recv_buffer[num_processed_bytes];

        /* when we get this cta_id_x it means the kernel has completed
         */
        if (ma->cta_id_x == -1) {
          recv_thread_receiving = false;
          break;
        }

        fprintf(ctx_resultsFile[ctx], "%d ", ma->cta_id_x);
        fprintf(ctx_resultsFile[ctx], "%d ", ma->cta_id_y);
        fprintf(ctx_resultsFile[ctx], "%d ", ma->cta_id_z);
        fprintf(ctx_resultsFile[ctx], "%d ", ma->warpid_tb);
        if (print_core_id) {
          fprintf(ctx_resultsFile[ctx], "%d ", ma->sm_id);
          fprintf(ctx_resultsFile[ctx], "%d ", ma->warpid_sm);
        }
        if (lineinfo) {
          fprintf(ctx_resultsFile[ctx], "%d ", ma->line_num);
        }
        fprintf(ctx_resultsFile[ctx], "%04x ", ma->vpc); // Print the virtual PC
        fprintf(ctx_resultsFile[ctx], "%08x ", ma->active_mask & ma->predicate_mask);
        if (ma->GPRDst >= 0) {
          fprintf(ctx_resultsFile[ctx], "1 ");
          fprintf(ctx_resultsFile[ctx], "R%d ", ma->GPRDst);
        } else
          fprintf(ctx_resultsFile[ctx], "0 ");

        // Print the opcode.
        fprintf(ctx_resultsFile[ctx], "%s ", id_to_opcode_map[ma->opcode_id].c_str());
        unsigned src_count = 0;
        for (int s = 0; s < MAX_SRC; s++) // GPR srcs count.
          if (ma->GPRSrcs[s] >= 0)
            src_count++;
        fprintf(ctx_resultsFile[ctx], "%d ", src_count);

        for (int s = 0; s < MAX_SRC; s++) // GPR srcs.
          if (ma->GPRSrcs[s] >= 0)
            fprintf(ctx_resultsFile[ctx], "R%d ", ma->GPRSrcs[s]);

        // print addresses
        std::bitset<32> mask(ma->active_mask & ma->predicate_mask);
        if (ma->is_mem) {
          std::istringstream iss(id_to_opcode_map[ma->opcode_id]);
          std::vector<std::string> tokens;
          std::string token;
          while (std::getline(iss, token, '.')) {
            if (!token.empty())
              tokens.push_back(token);
          }
          fprintf(ctx_resultsFile[ctx], "%d ", get_datawidth_from_opcode(tokens));

          bool base_stride_success = false;
          uint64_t base_addr = 0;
          int stride = 0;
          std::vector<long long> deltas;

          if (enable_compress) {
            // try base+stride format
            base_stride_success =
                base_stride_compress(ma->addrs, mask, base_addr, stride);
            if (!base_stride_success) {
              // if base+stride fails, try base+delta format
              base_delta_compress(ma->addrs, mask, base_addr, deltas);
            }
          }

          if (base_stride_success && enable_compress) {
            // base + stride format
            fprintf(ctx_resultsFile[ctx], "%u 0x%llx %d ", address_format::base_stride,
                    base_addr, stride);
          } else if (!base_stride_success && enable_compress) {
            // base + delta format
            fprintf(ctx_resultsFile[ctx], "%u 0x%llx ", address_format::base_delta,
                    base_addr);
            for (int s = 0; s < deltas.size(); s++) {
              fprintf(ctx_resultsFile[ctx], "%lld ", deltas[s]);
            }
          } else {
            // list all the addresses
            fprintf(ctx_resultsFile[ctx], "%u ", address_format::list_all);
            for (int s = 0; s < 32; s++) {
              if (mask.test(s))
                fprintf(ctx_resultsFile[ctx], "0x%016lx ", ma->addrs[s]);
            }
          }
        } else {
          fprintf(ctx_resultsFile[ctx], "0 ");
        }

        // Print the immediate
        fprintf(ctx_resultsFile[ctx], "%d ", ma->imm);

        fprintf(ctx_resultsFile[ctx], "\n");

        num_processed_bytes += sizeof(inst_trace_t);
      }
    }
  }
  free(recv_buffer);
  return NULL;
}

void nvbit_tool_init(CUcontext ctx) {
  recv_thread_started = true;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
  pthread_create(&recv_thread, NULL, recv_thread_fun, ctx);
}

void nvbit_at_ctx_term(CUcontext ctx) {
  if (recv_thread_started) {
    recv_thread_started = false;
    pthread_join(recv_thread, NULL);
  }
}

void nvbit_at_ctx_init(CUcontext ctx)
{
  // Everytime we init a context, add the foldername and kernelid to the set
  char buffer[2048];
  sprintf(buffer, "kernelslist_ctx_0x%lx", ctx);
  std::string tmp_kernelslist = user_folder + "/traces/" + buffer;
  ctx_kernelslist[ctx] = tmp_kernelslist;
  sprintf(buffer, "stats_ctx_0x%lx", ctx);
  std::string tmp_stats = user_folder + "/traces/" + buffer;
  ctx_stats_location[ctx] = tmp_stats;
  ctx_kernelid[ctx] = 1;
}