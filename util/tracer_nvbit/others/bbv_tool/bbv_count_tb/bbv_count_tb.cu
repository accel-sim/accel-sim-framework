/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Author: Oreste Villa, ovilla@nvidia.com - 2018 */

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <map>

#include <iostream>
using namespace std;
/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* provide some __device__ functions */
#include "utils/utils.h"

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_instrs = 0;

/* kernel instruction counter, updated by the GPU threads */
__managed__ uint64_t counter = 0;

/* pointer to memory location containing BBVs */
__managed__ int *bbv;

// Total threads of the kernel being launched
unsigned int tot_blocks = 0;

// Unique kernel ID
unsigned int kid = 0;

// Bool First allocation
bool first = true;
std::string fname = "bb_log_";

// Total number of basic blocks to keep track of
__managed__ unsigned int basic_blocks = 0;
std::map <std::string, int> kbb_map;
std::map <std::string, std::vector<int>> kbb_insns;

/* global control variables for this tool */
uint32_t ker_begin_interval = 0;
uint32_t ker_end_interval = UINT32_MAX;
int verbose = 1;
int count_warp_level = 1;
int exclude_pred_off = 0;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

/* instrumentation function that we want to inject, please note the use of
 * 1. "extern "C" __device__ __noinline__" to prevent code elimination by the
 * compiler.
 * 2. NVBIT_EXPORT_FUNC(count_instrs) to notify nvbit the name of the function
 * we want to inject. This name must match exactly the function name */
extern "C" __device__ __noinline__ void count_instrs(int num_instrs,
                                                     int count_warp_level,
                                                     int bb) {
    // Get the global warp id to update the bbv
    int global_wid = get_global_warp_id();
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot(1);
    /* each thread will get a lane id (get_lane_id is in utils/utils.h) */
    const int laneid = get_laneid();
    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;
    /* count all the active thread */
    const int num_threads = __popc(active_mask);
    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        // Index based upon the bb param and the threadblock number
        int global_tb = blockIdx.y * gridDim.x + blockIdx.x;

        // Must use atomic here to avoid coherence problems
        atomicAdd(&bbv[global_tb * basic_blocks + bb], num_threads);
    }
}
NVBIT_EXPORT_FUNC(count_instrs);

extern "C" __device__ __noinline__ void count_pred_off(int predicate,
                                                       int count_warp_level) {
    const int active_mask = __ballot(1);

    const int laneid = get_laneid();

    const int first_laneid = __ffs(active_mask) - 1;

    const int predicate_mask = __ballot(predicate);

    const int mask_off = active_mask ^ predicate_mask;

    const int num_threads_off = __popc(mask_off);
    if (first_laneid == laneid) {
        if (count_warp_level) {
            /* if the predicate mask was off we reduce the count of 1 */
            if (predicate_mask == 0)
                atomicAdd((unsigned long long *)&counter, -1);
        } else {
            atomicAdd((unsigned long long *)&counter, -num_threads_off);
        }
    }
}
NVBIT_EXPORT_FUNC(count_pred_off)

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We
 * typically do initializations in this call. In this case for instance we get
 * some environment variables values which we use as input arguments to the tool
 */
void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(ker_begin_interval, "KERNEL_BEGIN", 0,
                "Beginning of the kernel launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(
        ker_end_interval, "KERNEL_END", UINT32_MAX,
        "End of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
                "Count warp level or thread level instructions");
    GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 0,
                "Exclude predicated off instruction from count");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

/* nvbit_at_function_first_load() is executed every time a function is loaded
 * for the first time. Inside this call-back we typically get the vector of SASS
 * instructions composing the loaded CUfunction. We can iterate on this vector
 * and insert call to instrumentation functions before or after each one of
 * them. */
void nvbit_at_function_first_load(CUcontext ctx, CUfunction func) {

    /* Get the static control flow graph of instruction */
    const CFG_t &cfg = nvbit_get_CFG(ctx, func);
    if (cfg.is_degenerate) {
        printf(
            "Warning: Function %s is degenerated, we can't compute basic "
            "blocks statically",
            nvbit_get_func_name(ctx, func));
    }

    if (verbose) {
        printf("Function %s\n", nvbit_get_func_name(ctx, func));
        /* print */
        int cnt = 0;
        for (auto &bb : cfg.bbs) {
            printf("Basic block id %d - num instructions %ld\n", cnt++,
                   bb->instrs.size());
            for (auto &i : bb->instrs) {
                i->print(" ");
            }
        }
    }

    if (0) {
        printf("inspecting %s - number basic blocks %ld\n",
               nvbit_get_func_name(ctx, func), cfg.bbs.size());
    }

    /* Iterate on basic block and inject the first instruction */
    int local_bb = 0;
    for (auto &bb : cfg.bbs) {
        Instr *i = bb->instrs[0];
        /* inject device function */
        nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
        /* add size of basic block in number of instruction */
        nvbit_add_call_arg_const_val32(i, bb->instrs.size());
        /* add count warp level option */
        nvbit_add_call_arg_const_val32(i, count_warp_level);
        /* add basic block number */
        nvbit_add_call_arg_const_val32(i, local_bb++);
        if (verbose) {
            i->print("Inject count_instr before - ");
        }
    }

    // First time seeing the kernel, allocate space here too
    kbb_map.insert(std::pair<std::string,int>(nvbit_get_func_name(ctx, func), cfg.bbs.size()));

    // Add instruction counts to a vector and another map
    std::vector<int> i_counts;
    for (auto &bb : cfg.bbs) {
        i_counts.push_back(bb->instrs.size());
    }
    kbb_insns.insert(std::pair<std::string,std::vector<int>>(nvbit_get_func_name(ctx, func), i_counts));
    int *bbs;
    basic_blocks = cfg.bbs.size();
    cudaMallocManaged(&bbs, (tot_blocks) * (basic_blocks) * sizeof(int));

    // Set our __managed__ pointer to our narrowly allocated region
    bbv = bbs;
    for(unsigned int i = 0; i < (tot_blocks) * (basic_blocks); i++){
        bbv[i] = 0;
    }

    if (exclude_pred_off) {
        /* iterate on instructions */
        for (auto i : nvbit_get_instrs(ctx, func)) {
            /* inject only if instruction has predicate */
            if (i->hasPred()) {
                /* inject function */
                nvbit_insert_call(i, "count_pred_off", IPOINT_BEFORE);
                /* add predicate as argument */
                nvbit_add_call_arg_pred_val(i);
                /* add count warp level option */
                nvbit_add_call_arg_const_val32(i, count_warp_level);
                if (verbose) {
                    i->print("Inject count_instr before - ");
                }
            }
        }
    }
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel) {
        /* cast params to cuLaunch_params since if we are here we know these are
         * the right parameters type */
        cuLaunch_params *p = (cuLaunch_params *)params;

        if (!is_exit) {
            /* if we are entering in a kernel launch:
             * 1. Lock the mutex to prevent multiple kernels to run concurrently
             * (overriding the counter) in case the user application does that
             * 2. Select if we want to run the instrumented or original
             * version of the kernel
             * 3. Reset the kernel instruction counter */

            pthread_mutex_lock(&mutex);

            // Get the launch parameters to we can narrowly allocate memory
            cuLaunchKernel_params_st *p_test = (cuLaunchKernel_params_st *)params;

            // Only look at 2D kernels (common case, might change if 3D ones exist)
            unsigned int gx = p_test->gridDimX;
            unsigned int gy = p_test->gridDimY;

            // Set global number of threablocks
            tot_blocks = gx * gy;

            // Allocate some space for each warp
            if(first){
                first = false;
            }else{
                cudaFree(bbv);
            }

            if (kernel_id >= ker_begin_interval &&
                kernel_id < ker_end_interval) {
                nvbit_enable_instrumented(ctx, p->f, true);
            } else {
                nvbit_enable_instrumented(ctx, p->f, false);
            }
            counter = 0;

            // Allocate space here if we have called the kernel before
            auto it = kbb_map.find(nvbit_get_func_name(ctx, p->f));
            if(it != kbb_map.end()){
                basic_blocks = it->second;
                int *bbs;
                cudaMallocManaged(&bbs, tot_blocks * (basic_blocks) * sizeof(int));
                // Set our __managed__ pointer to our narrowly allocated region
                bbv = bbs;
                for(unsigned int i = 0; i < tot_blocks * (basic_blocks); i++){
                    bbv[i] = 0;
                }
            }



        } else {
            /* if we are exiting a kernel launch:
             * 1. Wait until the kernel is completed using
             * cudaDeviceSynchronize()
             * 2. Get number of thread blocks in the kernel
             * 3. Print the thread instruction counters
             * 4. Release the lock*/
            CUDA_SAFECALL(cudaDeviceSynchronize());

            // Get the launch parameters to we can narrowly allocate memory
            cuLaunchKernel_params_st *p_test = (cuLaunchKernel_params_st *)params;

            auto it = kbb_insns.find(nvbit_get_func_name(ctx, p->f));
            std::vector<int> test = it->second;
            FILE *f = fopen((fname + std::to_string(kid) + ".txt").c_str(), "w+");
            kid++;
            fprintf(f, "%s\n", nvbit_get_func_name(ctx, p->f));
            // For each basic block vector
            for(unsigned int i = 0; i < tot_blocks; i++){
                for(unsigned int j = 0; j < (basic_blocks); j++){
                    fprintf(f, "%d ", bbv[i * (basic_blocks) + j] * test[j] );
                }
                fprintf(f, "\n");
            }
            fclose(f);

            pthread_mutex_unlock(&mutex);
        }
    }
}
