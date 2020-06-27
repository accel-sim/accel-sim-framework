/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#include <algorithm>
#include <assert.h>
#include <bitset>
#include <inttypes.h>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>
/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"

#define TRACER_VERSION "1.2"

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

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* kernel instruction counter, updated by the GPU */
uint64_t dynamic_kernel_limit_start =
    0;                                 // 0 means start from the begging kernel
uint64_t dynamic_kernel_limit_end = 0; // 0 means no limit

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 1,
              "Exclude predicated off instruction from count");
  GET_VAR_INT(dynamic_kernel_limit_end, "DYNAMIC_KERNEL_LIMIT_END", 0,
              "Limit of the number kernel to be printed, 0 means no limit");
  GET_VAR_INT(dynamic_kernel_limit_start, "DYNAMIC_KERNEL_LIMIT_START", 0,
              "start to report kernel from this kernel id, 0 means starts from "
              "the begging, i.e. first kernel");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(enable_compress, "TOOL_COMPRESS", 1, "Enable traces compression");
  GET_VAR_INT(print_core_id, "TOOL_TRACE_CORE", 0,
              "write the core id in the traces");
  std::string pad(100, '-');
  printf("%s\n", pad.c_str());
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
             nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f), true);
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }

      if (verbose) {
        instr->printDecoded();
      }

      if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = instr->getOpcode();
      }

      int opcode_id = opcode_to_id_map[instr->getOpcode()];

      /* insert call to the instrumentation function with its
       * arguments */
      nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);

      /* pass predicate value */
      nvbit_add_call_arg_pred_val(instr);

      /* send opcode and pc */
      nvbit_add_call_arg_const_val32(instr, opcode_id);
      nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());

      /* check all operands
         For now, we ignore constant, TEX, predicates and unified registers.
         We only report regisers */
      int src_oprd[MAX_SRC];
      int srcNum = 0;
      int dst_oprd = -1;
      int mem_oper_idx = -1;

      // find dst reg and handle the special case if the oprd[0] is mem (e.g.
      // store and RED)
      if (instr->getNumOperands() > 0 &&
          instr->getOperand(0)->type == Instr::operandType::REG)
        dst_oprd = instr->getOperand(0)->u.reg.num;
      else if (instr->getNumOperands() > 0 &&
               instr->getOperand(0)->type == Instr::operandType::MREF) {
        src_oprd[0] = instr->getOperand(0)->u.mref.ra_num;
        mem_oper_idx = 0;
        srcNum++;
      } else
        src_oprd[0] = -1;

      // find src regs and mem
      assert(instr->getNumOperands() <= MAX_SRC);
      for (int i = 1; i < MAX_SRC; i++) {
        if (i < instr->getNumOperands()) {
          const Instr::operand_t *op = instr->getOperand(i);
          if (op->type == Instr::operandType::MREF) {
            // mem is found
            src_oprd[i] = instr->getOperand(i)->u.mref.ra_num;
            assert(mem_oper_idx == -1); // ensure one memory operand per inst
            mem_oper_idx = i;
          } else if (op->type == Instr::operandType::REG)
            src_oprd[i] = instr->getOperand(i)->u.reg.num;
          else
            src_oprd[i] = -1;

          srcNum++;
        } else
          src_oprd[i] = -1;
      }

      /* mem addresses info */
      if (mem_oper_idx >= 0) {
        const Instr::operand_t *mem_op = instr->getOperand(mem_oper_idx);

        nvbit_add_call_arg_const_val32(instr, 1);
        if (instr->isExtended()) {
          nvbit_add_call_arg_reg_val(instr, (int)mem_op->u.mref.ra_num + 1);
        } else {
          nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
        }
        nvbit_add_call_arg_reg_val(instr, (int)mem_op->u.mref.ra_num);
        nvbit_add_call_arg_const_val32(instr, (int)mem_op->u.mref.imm);
        nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());
      } else {
        nvbit_add_call_arg_const_val32(instr, 0);
        nvbit_add_call_arg_const_val32(instr, -1);
        nvbit_add_call_arg_const_val32(instr, -1);
        nvbit_add_call_arg_const_val32(instr, -1);
        nvbit_add_call_arg_const_val32(instr, -1);
      }

      /* reg info */
      nvbit_add_call_arg_const_val32(instr, dst_oprd);
      for (int i = 0; i < MAX_SRC; i++) {
        nvbit_add_call_arg_const_val32(instr, src_oprd[i]);
      }
      nvbit_add_call_arg_const_val32(instr, srcNum);

      /* add pointer to channel_dev and other counters*/
      nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
      nvbit_add_call_arg_const_val64(instr,
                                     (uint64_t)&total_dynamic_instr_counter);
      nvbit_add_call_arg_const_val64(instr,
                                     (uint64_t)&reported_dynamic_instr_counter);
      nvbit_add_call_arg_const_val64(instr, (uint64_t)&stop_report);
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

    if (mkdir("traces", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
      if (errno == EEXIST) {
        // alredy exists
      } else {
        // something else
        std::cout << "cannot create folder error:" << strerror(errno)
                  << std::endl;
        return;
      }
    }

    if (!dynamic_kernel_limit_start || dynamic_kernel_limit_start == 1)
      stop_report = false;
    else
      stop_report = true;

    kernelsFile = fopen("./traces/kernelslist", "w");
    statsFile = fopen("./traces/stats.csv", "w");
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
      kernelsFile = fopen("./traces/kernelslist", "a");
      sprintf(buffer, "MemcpyHtoD,0x%016lx,%lld", p->dstDevice, p->ByteCount);
      fprintf(kernelsFile, buffer);
      fprintf(kernelsFile, "\n");
      fclose(kernelsFile);
    }

  } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
             cbid == API_CUDA_cuLaunchKernel) {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

    if (!is_exit) {

      if (dynamic_kernel_limit_start && kernelid == dynamic_kernel_limit_start)
        stop_report = false;

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

      nvbit_enable_instrumented(ctx, p->f, true);

      char buffer[1024];
      sprintf(buffer, "./traces/kernel-%d.trace", kernelid);

      if (!stop_report) {
        resultsFile = fopen(buffer, "w");

        printf("Writing results to %s\n", buffer);

        fprintf(resultsFile, "-kernel name = %s\n",
                nvbit_get_func_name(ctx, p->f, true));
        fprintf(resultsFile, "-kernel id = %d\n", kernelid);
        fprintf(resultsFile, "-grid dim = (%d,%d,%d)\n", p->gridDimX,
                p->gridDimY, p->gridDimZ);
        fprintf(resultsFile, "-block dim = (%d,%d,%d)\n", p->blockDimX,
                p->blockDimY, p->blockDimZ);
        fprintf(resultsFile, "-shmem = %d\n",
                shmem_static_nbytes + p->sharedMemBytes);
        fprintf(resultsFile, "-nregs = %d\n", nregs);
        fprintf(resultsFile, "-binary version = %d\n", binary_version);
        fprintf(resultsFile, "-cuda stream id = %d\n", (uint64_t)p->hStream);
        fprintf(resultsFile, "-shmem base_addr = 0x%016lx\n",
                (uint64_t)nvbit_get_shmem_base_addr(ctx));
        fprintf(resultsFile, "-local mem base_addr = 0x%016lx\n",
                (uint64_t)nvbit_get_local_mem_base_addr(ctx));
        fprintf(resultsFile, "-nvbit version = %s\n", NVBIT_VERSION);
        fprintf(resultsFile, "-accelsim tracer version = %s\n", TRACER_VERSION);
        fprintf(resultsFile, "\n");

        fprintf(resultsFile,
                "#traces format = threadblock_x threadblock_y threadblock_z "
                "warpid_tb PC mask dest_num [reg_dests] opcode src_num "
                "[reg_srcs] mem_width [adrrescompress?] [mem_addresses]\n");
        fprintf(resultsFile, "\n");
      }

      kernelsFile = fopen("./traces/kernelslist", "a");
      sprintf(buffer, "kernel-%d.trace", kernelid);
      if (!stop_report) {
        fprintf(kernelsFile, buffer);
        fprintf(kernelsFile, "\n");
      }
      fclose(kernelsFile);

      statsFile = fopen("./traces/stats.csv", "a");
      unsigned blocks = p->gridDimX * p->gridDimY * p->gridDimZ;
      unsigned threads = p->blockDimX * p->blockDimY * p->blockDimZ;

      fprintf(statsFile, "%s, %s, %d, %d, %d, %d, %d, %d, %d, %d, ", buffer,
              nvbit_get_func_name(ctx, p->f, true), p->gridDimX, p->gridDimY,
              p->gridDimZ, blocks, p->blockDimX, p->blockDimY, p->blockDimZ,
              threads);

      fclose(statsFile);

      kernelid++;
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

      statsFile = fopen("./traces/stats.csv", "a");
      fprintf(statsFile, "%d,%d", total_insts_per_kernel,
              reported_insts_per_kernel);
      fprintf(statsFile, "\n");
      fclose(statsFile);

      if (!stop_report)
        fclose(resultsFile);

      if (dynamic_kernel_limit_end && kernelid > dynamic_kernel_limit_end)
        stop_report = true;
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

void *recv_thread_fun(void *) {
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

        fprintf(resultsFile, "%d ", ma->cta_id_x);
        fprintf(resultsFile, "%d ", ma->cta_id_y);
        fprintf(resultsFile, "%d ", ma->cta_id_z);
        fprintf(resultsFile, "%d ", ma->warpid_tb);
        if (print_core_id) {
          fprintf(resultsFile, "%d ", ma->sm_id);
          fprintf(resultsFile, "%d ", ma->warpid_sm);
        }
        fprintf(resultsFile, "%04x ", ma->vpc); // Print the virtual PC
        fprintf(resultsFile, "%08x ", ma->active_mask & ma->predicate_mask);
        if (ma->GPRDst >= 0) {
          fprintf(resultsFile, "1 ");
          fprintf(resultsFile, "R%d ", ma->GPRDst);
        } else
          fprintf(resultsFile, "0 ");

        // Print the opcode.
        fprintf(resultsFile, "%s ", id_to_opcode_map[ma->opcode_id].c_str());
        unsigned src_count = 0;
        for (int s = 0; s < MAX_SRC; s++) // GPR srcs count.
          if (ma->GPRSrcs[s] >= 0)
            src_count++;
        fprintf(resultsFile, "%d ", src_count);

        for (int s = 0; s < MAX_SRC; s++) // GPR srcs.
          if (ma->GPRSrcs[s] >= 0)
            fprintf(resultsFile, "R%d ", ma->GPRSrcs[s]);

        // print addresses
        std::bitset<32> mask(ma->active_mask);
        if (ma->is_mem) {
          // fprintf(resultsFile, "%d ", ma->width);
          std::istringstream iss(id_to_opcode_map[ma->opcode_id]);
          std::vector<std::string> tokens;
          std::string token;
          while (std::getline(iss, token, '.')) {
            if (!token.empty())
              tokens.push_back(token);
          }
          fprintf(resultsFile, "%d ", get_datawidth_from_opcode(tokens));

          // calulcate teh difference between addresses
          // write cosnsctive addresses with constant stride in a more
          // compressed way (i.e. start adress and stride)
          int stride = 0;
          bool const_stride = true;
          bool first_bit1_found = false;
          bool last_bit1_found = false;
          uint64_t addr = 0;
          for (int s = 0; s < 32; s++) {
            if (mask.test(s) && !first_bit1_found) {
              first_bit1_found = true;
              addr = ma->addrs[s];
              if (s < 31 && mask.test(s + 1))
                stride = ma->addrs[s + 1] - ma->addrs[s];
              else {
                const_stride = false;
                break;
              }
            } else if (first_bit1_found && !last_bit1_found) {
              if (mask.test(s)) {
                if (stride != ma->addrs[s] - ma->addrs[s - 1]) {
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

          if (const_stride && enable_compress) {
            fprintf(resultsFile, "1 0x%016lx %d ", addr, stride);
          } else {
            // list all the adresses
            fprintf(resultsFile, "0 ");
            for (int s = 0; s < 32; s++) {
              if (mask.test(s))
                fprintf(resultsFile, "0x%016lx ", ma->addrs[s]);
            }
          }
        } else {
          fprintf(resultsFile, "0 ");
        }

        fprintf(resultsFile, "\n");

        num_processed_bytes += sizeof(inst_trace_t);
      }
    }
  }
  free(recv_buffer);
  return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
  recv_thread_started = true;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
  pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);
}

void nvbit_at_ctx_term(CUcontext ctx) {
  if (recv_thread_started) {
    recv_thread_started = false;
    pthread_join(recv_thread, NULL);
  }
}
