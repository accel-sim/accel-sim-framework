/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */


#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <map>
#include <inttypes.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <bitset>
#include <sys/stat.h>
#include <sstream>
#include <algorithm>
#include <iterator>
/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* for _cuda_safe and GET_VAR* macros */
//#include "macros.h"

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
static __managed__ uint64_t total_dynamic_instr_counter = 0;
static __managed__ uint64_t reported_dynamic_instr_counter = 0;
static __managed__ uint64_t dynamic_instr_limit = 0;
static __managed__ bool stop_report = false;
uint64_t dynamic_instr_limit_input = 0;  //0 means no limit
uint64_t dynamic_kernel_limit_input = 0;  //0 means no limit

#define MAX_SRC 4
/* information collected in the instrumentation function */
typedef struct {
	int cta_id_x;
	int cta_id_y;
	int cta_id_z;
	int warpid_tb;
	int warpid_sm;
	int sm_id;
	int opcode_id;
	uint64_t addrs[32];
	uint32_t vpc;
	bool is_mem;
	int32_t GPRDst;
	int32_t GPRSrcs[MAX_SRC];
	int32_t numSrcs;
	int32_t width;
	uint32_t active_mask;
	uint32_t predicate_mask;

} mem_access_t;

/* Instrumentation function that we want to inject, please note the use of
 * 1. extern "C" __device__ __noinline__
 *    To prevent "dead"-code elimination by the compiler.
 * 2. NVBIT_EXPORT_FUNC(dev_func)
 *    To notify nvbit the name of the function we want to inject.
 *    This name must match exactly the function name.
 */
extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id, int32_t vpc,
		uint32_t reg_high,
		uint32_t reg_low,
		int32_t imm,
		int32_t desReg, int32_t srcReg1, int32_t srcReg2, int32_t srcReg3, int32_t srcReg4, int srcNum, int32_t width) {
	//if (!pred) {
	//	return;
	//}

	const int active_mask = __ballot(1);
	const int predicate_mask = __ballot(pred);
	const int laneid = get_laneid();
	const int first_laneid = __ffs(active_mask) - 1;

	if ((dynamic_instr_limit && total_dynamic_instr_counter >= dynamic_instr_limit) || stop_report) {
		if (first_laneid == laneid) {
			atomicAdd((unsigned long long*)&total_dynamic_instr_counter, 1);
			return;
		}
	}

	mem_access_t ma;

	/* collect memory address information */
	int64_t base_addr = (((uint64_t)reg_high) << 32) | ((uint64_t)reg_low);
	uint64_t addr = base_addr + imm;
	for (int i = 0; i < 32; i++) {
		ma.addrs[i] = __shfl(addr, i);
	}

	int4 cta = get_ctaid();
	int uniqe_threadId = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	ma.warpid_tb = uniqe_threadId/32;

	ma.cta_id_x = cta.x;
	ma.cta_id_y = cta.y;
	ma.cta_id_z = cta.z;
	ma.warpid_sm = get_warpid();
	ma.opcode_id = opcode_id;
	ma.is_mem = true;
	ma.vpc = vpc;
	ma.width = width;
	ma.GPRDst = desReg;
	ma.GPRSrcs[0] = srcReg1;
	ma.GPRSrcs[1] = srcReg2;
	ma.GPRSrcs[2] = srcReg3;
	ma.GPRSrcs[3] = srcReg4;
	ma.numSrcs = srcNum;
	ma.active_mask = active_mask;
	ma.predicate_mask = predicate_mask;
	ma.sm_id =  get_smid();

	/* first active lane pushes information on the channel */
	if (first_laneid == laneid) {
		channel_dev.push(&ma, sizeof(mem_access_t));
		atomicAdd((unsigned long long*)&total_dynamic_instr_counter, 1);
		atomicAdd((unsigned long long*)&reported_dynamic_instr_counter, 1);
	}
}
NVBIT_EXPORT_FUNC(instrument_mem);


extern "C" __device__ __noinline__ void instrument_inst(int pred, int opcode_id,
		uint32_t vpc, int desReg, int srcReg1, int srcReg2, int srcReg3, int srcReg4, int srcNum) {
	//if (!pred) {
	//	return;
	//}

	int active_mask = __ballot(1);
	const int predicate_mask = __ballot(pred);
	const int laneid = get_laneid();
	const int first_laneid = __ffs(active_mask) - 1;

	if ((dynamic_instr_limit && total_dynamic_instr_counter >= dynamic_instr_limit) || stop_report) {
		if (first_laneid == laneid) {
			atomicAdd((unsigned long long*)&total_dynamic_instr_counter, 1);
			return;
		}
	}


	mem_access_t ma;

	int4 cta = get_ctaid();
	int uniqe_threadId = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	ma.warpid_tb = uniqe_threadId/32;

	ma.cta_id_x = cta.x;
	ma.cta_id_y = cta.y;
	ma.cta_id_z = cta.z;
	ma.warpid_sm = get_warpid();
	ma.opcode_id = opcode_id;
	ma.is_mem = false;
	ma.vpc = vpc;

	ma.GPRDst = desReg;
	ma.numSrcs = srcNum;    //this is the total src number including the register and others
	ma.GPRSrcs[0] = srcReg1;
	ma.GPRSrcs[1] = srcReg2;
	ma.GPRSrcs[2] = srcReg3;
	ma.GPRSrcs[3] = srcReg4;

	ma.active_mask = active_mask;
	ma.predicate_mask = predicate_mask;
	ma.sm_id =  get_smid();

	/* first active lane pushes information on the channel */
	if (first_laneid == laneid) {
		channel_dev.push(&ma, sizeof(mem_access_t));
		atomicAdd((unsigned long long*)&total_dynamic_instr_counter, 1);
		atomicAdd((unsigned long long*)&reported_dynamic_instr_counter, 1);
	}
}

NVBIT_EXPORT_FUNC(instrument_inst);

void nvbit_at_init() {
	setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
	GET_VAR_INT(
			instr_begin_interval, "INSTR_BEGIN", 0,
			"Beginning of the instruction interval where to apply instrumentation");
	GET_VAR_INT(
			instr_end_interval, "INSTR_END", UINT32_MAX,
			"End of the instruction interval where to apply instrumentation");
	GET_VAR_INT(
			exclude_pred_off, "EXCLUDE_PRED_OFF", 1,
			"Exclude predicated off instruction from count");
	GET_VAR_LONG(
			dynamic_instr_limit_input, "DYNAMIC_INSTR_LIMIT", 0,
			"Limit of the number instructions to be printed, 0 means no limit");
	GET_VAR_INT(
			dynamic_kernel_limit_input, "DYNAMIC_KERNEL_LIMIT", 0,
			"Limit of the number kernel to be printed, 0 means no limit");
	GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
	GET_VAR_INT(enable_compress, "TOOL_COMPRESS", 1, "Enable traces compression");
	GET_VAR_INT(print_core_id, "TOOL_TRACE_CORE", 0, "write the core id in the traces");
	std::string pad(100, '-');
	printf("%s\n", pad.c_str());
}

/* instrument each memory instruction adding a call to the above instrumentation
 * function */
void nvbit_at_function_first_load(CUcontext ctx, CUfunction f) {

	dynamic_instr_limit = dynamic_instr_limit_input;

	const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
	if (verbose) {
		printf("Inspecting function %s at address 0x%lx\n",
				nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f), true);
	}

	uint32_t cnt = 0;
	/* iterate on all the static instructions in the function */
	for (auto instr : instrs) {
		if (cnt < instr_begin_interval || cnt >= instr_end_interval ) {
			cnt++;
			continue;
		}
		//if (verbose) {
		instr->printDecoded();
		//}

		if (opcode_to_id_map.find(instr->getOpcode()) ==
				opcode_to_id_map.end()) {
			int opcode_id = opcode_to_id_map.size();
			opcode_to_id_map[instr->getOpcode()] = opcode_id;
			id_to_opcode_map[opcode_id] = instr->getOpcode();
		}

		int opcode_id = opcode_to_id_map[instr->getOpcode()];

		//TO DO: handle generic and TEX memory space
		if(instr->isLoad() && !instr->isStore() && instr->getMemOpType() != Instr::CONSTANT) {   //Mem load inst //ignore constant for now
			assert(instr->getNumOperands() == 2);

			/* get the operand */
			const Instr::operand_t *dst = instr->getOperand(0);
			const Instr::operand_t *src = instr->getOperand(1);

			assert(dst->type == Instr::REG);
			assert(src->type == Instr::MREF);

			/* insert call to the instrumentation function with its
			 * arguments */
			nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);

			/* pass predicate value */
			nvbit_add_call_arg_pred_val(instr);

			nvbit_add_call_arg_const_val32(instr, opcode_id);
			nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());
			if (instr->isExtended()) {
				nvbit_add_call_arg_reg_val(instr, (int)src->value[0] + 1);
			} else {
				nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
			}
			nvbit_add_call_arg_reg_val(instr, (int)src->value[0]);
			nvbit_add_call_arg_const_val32(instr, (int)src->value[1]);
			nvbit_add_call_arg_const_val32(instr, (int)dst->value[0]);
			nvbit_add_call_arg_const_val32(instr, (int)src->value[0]);
			nvbit_add_call_arg_const_val32(instr, -1);
			nvbit_add_call_arg_const_val32(instr, -1);
			nvbit_add_call_arg_const_val32(instr, -1);
			nvbit_add_call_arg_const_val32(instr, 1);
			nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());

		}
		else if(instr->isStore() && !instr->isLoad() && instr->getMemOpType() != Instr::CONSTANT) {   //Mem store inst //ignore constant for now
			assert(instr->getNumOperands() == 2);

			/* get the operand */
			const Instr::operand_t *dst = instr->getOperand(0);
			const Instr::operand_t *src = instr->getOperand(1);

			assert(dst->type == Instr::MREF);
			assert(src->type == Instr::REG);

			/* insert call to the instrumentation function with its
			 * arguments */
			nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
			/* pass predicate value */
			nvbit_add_call_arg_pred_val(instr);
			nvbit_add_call_arg_const_val32(instr, opcode_id);
			nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());
			if (instr->isExtended()) {
				nvbit_add_call_arg_reg_val(instr, (int)dst->value[0] + 1);
			} else {
				nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
			}
			nvbit_add_call_arg_reg_val(instr, (int)dst->value[0]);
			nvbit_add_call_arg_const_val32(instr, (int)dst->value[1]);
			nvbit_add_call_arg_const_val32(instr, -1);
			nvbit_add_call_arg_const_val32(instr, (int)dst->value[0]);
			nvbit_add_call_arg_const_val32(instr, (int)src->value[0]);
			nvbit_add_call_arg_const_val32(instr, -1);
			nvbit_add_call_arg_const_val32(instr, -1);
			nvbit_add_call_arg_const_val32(instr, 2);
			nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());
		}
		else if(instr->isLoad() && instr->isStore() && instr->getMemOpType() != Instr::CONSTANT) {   //if it is load and store i.e. atomic inst 
			/* find memory operand */
			int src_oprd[MAX_SRC+1];
			int srcNum=0;
			int mem_oper=-1;
			for (int i = 0; i < MAX_SRC+1; i++) {
				if(i < instr->getNumOperands()) {
					const Instr::operand_t *op = instr->getOperand(i);
					if (op->type == Instr::MREF){
						//mem is found
						src_oprd[i]=instr->getOperand(i)->value[0];
						mem_oper=i;
					}
					else if (op->type == Instr::REG)
						src_oprd[i]=instr->getOperand(i)->value[0];
					else
						src_oprd[i]=-1;

					srcNum++;
				}
				else
					src_oprd[i]=-1;
			}

			assert(mem_oper>=0);
			const Instr::operand_t *dst = instr->getOperand(mem_oper);

			//do not write to the reg if it is a memory, the case happens in case of RED inst
			if(mem_oper == 0)
				src_oprd[0]=-1;

			/* insert call to the instrumentation function with its
			 * arguments */
			nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
			/* pass predicate value */
			nvbit_add_call_arg_pred_val(instr);
			nvbit_add_call_arg_const_val32(instr, opcode_id);
			nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());
			if (instr->isExtended()) {
				nvbit_add_call_arg_reg_val(instr, (int)dst->value[0] + 1);
			} else {
				nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
			}
			nvbit_add_call_arg_reg_val(instr, (int)dst->value[0]);
			nvbit_add_call_arg_const_val32(instr, (int)dst->value[1]);

			for (int i = 0; i < MAX_SRC+1; i++) {
				nvbit_add_call_arg_const_val32(instr, src_oprd[i]);
			}
			nvbit_add_call_arg_const_val32(instr, srcNum);
			nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());

		}
		else    //Other ALU, FP, DP insts
		{

			nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);
			/* pass predicate value */
			nvbit_add_call_arg_pred_val(instr);
			nvbit_add_call_arg_const_val32(instr, opcode_id);
			nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());
			int srcNum = 0;
			for (int i = 0; i < MAX_SRC+1; i++) {
				/* get the operand "i" */
				if(i < instr->getNumOperands()) {
					const Instr::operand_t *op = instr->getOperand(i);
					if (op->type == Instr::REG)
						nvbit_add_call_arg_const_val32(instr, (int)op->value[0]);
					else
						nvbit_add_call_arg_const_val32(instr, -1);

					srcNum++;
				}
				else
					nvbit_add_call_arg_const_val32(instr, -1);
			}
			nvbit_add_call_arg_const_val32(instr, srcNum);
		}
		cnt++;
	}
}

__global__ void flush_channel() {
	/* push memory access with negative cta id to communicate the kernel is
	 * completed */
	mem_access_t ma;
	ma.cta_id_x = -1;
	channel_dev.push(&ma, sizeof(mem_access_t));

	/* flush channel */
	channel_dev.flush();
}

static FILE *resultsFile = NULL;
static FILE *kernelsFile= NULL;
static FILE *statsFile= NULL;
static int kernelid = 1;
static bool first_call = true;

unsigned old_total_insts = 0;
unsigned old_total_reported_insts = 0;


void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
		const char *name, void *params, CUresult *pStatus) {

	if (skip_flag) return;
	
			if (first_call == true) {

				first_call=false;

			if (mkdir("traces", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
				if( errno == EEXIST ) {
					// alredy exists
				} else {
					// something else
					std::cout << "cannot create folder error:" << strerror(errno) << std::endl;
					return;
				}
			}

				kernelsFile = fopen("./traces/kernelslist", "w");
				statsFile = fopen("./traces/stats.csv", "w");
				fprintf(statsFile, "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, #blocks, block_dimX, block_dimY, block_dimZ, #threads, total_insts, total_reported_insts\n");
			}

	if (cbid == API_CUDA_cuMemcpyHtoD_v2) {
		if (!is_exit) {
			cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;
			char buffer[1024];
			kernelsFile = fopen("./traces/kernelslist", "a");
			sprintf (buffer, "MemcpyHtoD,0x%016lx,%lld", p->dstDevice, p->ByteCount);
			if(!stop_report) {
				fprintf(kernelsFile, buffer);
				fprintf(kernelsFile, "\n");
			}
			fclose(kernelsFile);
		}
		
	}
	else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
			cbid == API_CUDA_cuLaunchKernel) {
		cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

		if (!is_exit) {




			int nregs;
			CUDA_SAFECALL(
					cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

			int shmem_static_nbytes;
			CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes,
					CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
					p->f));


			int binary_version;
			CUDA_SAFECALL(cuFuncGetAttribute(&binary_version,
					CU_FUNC_ATTRIBUTE_BINARY_VERSION,
					p->f));


			//std::string func_name(nvbit_get_func_name(ctx, p->f, true));
			//std::string::size_type end_pos = func_name.find('(');
			//if (end_pos != std::string::npos)
			//{
			// std::string::size_type pos = func_name.find('<');
			//if (pos != std::string::npos)
			//	end_pos = pos;

			//std::string::size_type start_pos = func_name.find(' ');
			//if (start_pos == std::string::npos)
			//	start_pos = 0;
			//else
			//	start_pos++;

			//func_name = func_name.substr(0, end_pos);
			//}

			char buffer[1024];
			sprintf (buffer, "./traces/kernel-%d.trace", kernelid);

			resultsFile = fopen(buffer, "w");

			printf("Writing results to %s\n", buffer);

			fprintf(resultsFile, "-kernel name = %s",  nvbit_get_func_name(ctx, p->f, true));
			fprintf(resultsFile, "\n");
			fprintf(resultsFile, "-kernel id = %d",  kernelid);
			fprintf(resultsFile, "\n");
			fprintf(resultsFile, "-grid dim = (%d,%d,%d)",  p->gridDimX, p->gridDimY, p->gridDimZ);
			fprintf(resultsFile, "\n");
			fprintf(resultsFile, "-block dim = (%d,%d,%d)",  p->blockDimX, p->blockDimY, p->blockDimZ);
			fprintf(resultsFile, "\n");
			fprintf(resultsFile, "-shmem = %d",   shmem_static_nbytes + p->sharedMemBytes);
			fprintf(resultsFile, "\n");
			fprintf(resultsFile, "-nregs = %d",   nregs);
			fprintf(resultsFile, "\n");
			fprintf(resultsFile, "-binary version = %d",   binary_version);
			fprintf(resultsFile, "\n");
			fprintf(resultsFile, "-cuda stream id = %d",  (uint64_t)p->hStream);
			fprintf(resultsFile, "\n\n");

			fprintf(resultsFile, "#traces format = threadblock_x threadblock_y threadblock_z warpid_tb PC mask dest_num reg_dests opcode src_num reg_srcs mem_width mem_addresses");
			fprintf(resultsFile, "\n");



				kernelsFile = fopen("./traces/kernelslist", "a");
				statsFile = fopen("./traces/stats.csv", "a");
	

			sprintf (buffer, "kernel-%d.trace", kernelid);
			if(!stop_report) {
				fprintf(kernelsFile, buffer);
				fprintf(kernelsFile, "\n");
			}
			fclose(kernelsFile);

			unsigned blocks = p->gridDimX * p->gridDimY * p->gridDimZ;
			unsigned threads = p->blockDimX * p->blockDimY* p->blockDimZ;
			fprintf(statsFile, "%s, %s, %d, %d, %d, %d, %d, %d, %d, %d, ", buffer,  nvbit_get_func_name(ctx, p->f, true),  p->gridDimX, p->gridDimY, p->gridDimZ, blocks,  p->blockDimX, p->blockDimY, p->blockDimZ, threads  );


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

			unsigned total_insts_per_kernel =  total_dynamic_instr_counter - old_total_insts;
			old_total_insts = total_dynamic_instr_counter;

			unsigned reported_insts_per_kernel =  reported_dynamic_instr_counter - old_total_reported_insts;
			old_total_reported_insts = reported_dynamic_instr_counter;

			fprintf(statsFile, "");
			fprintf(statsFile, "%d,%d",total_insts_per_kernel,reported_insts_per_kernel);
			fprintf(statsFile, "\n");

			if(dynamic_kernel_limit_input && kernelid > dynamic_kernel_limit_input)
				stop_report=true;

			fclose(resultsFile);
			fclose(statsFile);
		}
	}
}

bool is_number(const std::string& s)
{
	std::string::const_iterator it = s.begin();
	while (it != s.end() && std::isdigit(*it)) ++it;
	return !s.empty() && it == s.end();
}

unsigned get_datawidth_from_opcode(const std::vector<std::string>& opcode)
{
	for(unsigned i=0; i<opcode.size(); ++i) {
		if(is_number(opcode[i])){
			return (std::stoi(opcode[i],NULL)/8);
		}
		else if(opcode[i][0] == 'U' && is_number(opcode[i].substr(1))){
			//handle the U* case
			unsigned bits;
			sscanf(opcode[i].c_str(), "U%u",&bits);
			return bits/8;
		}
	}

	return 4;  //default is 4 bytes
}

void *recv_thread_fun(void *) {
	char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

	while (recv_thread_started) {
		uint32_t num_recv_bytes = 0;
		if (recv_thread_receiving &&
				(num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) >
		0) {
			uint32_t num_processed_bytes = 0;
			while (num_processed_bytes < num_recv_bytes) {
				mem_access_t *ma =
						(mem_access_t *)&recv_buffer[num_processed_bytes];

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
				if(print_core_id) {
					fprintf(resultsFile, "%d ", ma->sm_id);
					fprintf(resultsFile, "%d ", ma->warpid_sm);
				}
				fprintf(resultsFile, "%04x ", ma->vpc); // Print the virtual PC 
				fprintf(resultsFile, "%08x ", ma->active_mask & ma->predicate_mask);
				if(ma->GPRDst >= 0) {
					fprintf(resultsFile, "1 ");
					fprintf(resultsFile, "R%d ", ma->GPRDst);
				}
				else
					fprintf(resultsFile, "0 ");

				// Print the opcode.
				fprintf(resultsFile, "%s ", id_to_opcode_map[ma->opcode_id].c_str());
				unsigned src_count=0;
				for (int s = 0; s < MAX_SRC; s++)      // GPR srcs count.
					if(ma->GPRSrcs[s] >= 0)  src_count++;
				fprintf(resultsFile, "%d ", src_count);

				for (int s = 0; s < MAX_SRC; s++)      // GPR srcs.
					if(ma->GPRSrcs[s] >= 0)  fprintf(resultsFile, "R%d ", ma->GPRSrcs[s]);

				//print addresses
				std::bitset<32> mask(ma->active_mask);
				if(ma->is_mem) {
					//fprintf(resultsFile, "%d ", ma->width);
					std::istringstream iss(id_to_opcode_map[ma->opcode_id]);
					std::vector<std::string> tokens;
					std::string token;
					while (std::getline(iss, token, '.')) {
						if (!token.empty())
							tokens.push_back(token);
					}
					fprintf(resultsFile, "%d ", get_datawidth_from_opcode(tokens));


					//calulcate teh difference between addresses
					//write cosnsctive addresses with constant stride in a more compressed way (i.e. start adress and stride)
					int stride=0;
					bool const_stride=true;
					bool first_bit1_found=false;
					bool last_bit1_found=false;
					uint64_t addr=0;
					for (int s = 0; s < 32; s++) {
						if(mask.test(s) && !first_bit1_found) {
							first_bit1_found = true;
							addr=ma->addrs[s];
							if(s < 31 && mask.test(s+1))
								stride=ma->addrs[s+1]-ma->addrs[s];
							else {
								const_stride=false;
								break;
							}
						} else if(first_bit1_found && !last_bit1_found) {
							if(mask.test(s)) {
								if(stride != ma->addrs[s]-ma->addrs[s-1]) {
									const_stride=false;
									break;
								}
							} else
								last_bit1_found=true;
						} else if(last_bit1_found) {
							if(mask.test(s)){
								const_stride=false;
								break;
							}
						}
					}

					if(const_stride && enable_compress){
						fprintf(resultsFile, "1 0x%016lx %d ", addr, stride);
					} else {
						//list all the adresses
						fprintf(resultsFile, "0 ");
						for (int s = 0; s < 32; s++) {
							if(mask.test(s))
								fprintf(resultsFile, "0x%016lx ", ma->addrs[s]);
						}
					}
				}
				else
				{
					fprintf(resultsFile, "0 ");
				}

				fprintf(resultsFile, "\n");

				num_processed_bytes += sizeof(mem_access_t);
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





