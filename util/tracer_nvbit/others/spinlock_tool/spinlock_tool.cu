/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * A tool to identify spinlock section in the SASS kernel.
 * The idea is to run the tool twice and count the executed number for
 * each instruction. For non-deterministic part (aka spinlock in our GEMM app), 
 * the number of executed instructions would be different, thus we can identify
 * the spinlock section.
 *
 * TODO Some questions:
 * 1. How to link result with accel-sim tracer?
 * 2. What should be the flow of this tool?
 *   - Multiple phases of the tool. Assuming the app kernel launch is deterministic, we can run the tool twice to detect the spinlock section.
 *   - Spinlock detection phase.
 *     - For each kernel, record the instruction execution count and save to file with marking of kernel id and kernel name.
 *     - This histogram tool can be run multiple times to determine the spinlock section.
 *   - Accel-sim tracer tracing phase.
 *     - It will try to consume the spinlock phase data for spinlock section of each kernel (by kernel id)
 *     - If the spinlock section is found, it will mark and trace these instructions only once.
 * 3. Is it possible to replay kernel with NVBit? Probably not, and bit wacky. So we just launch the app multiple times.
 * 4. How to find the chain of the producer of the spinlock sections?
 *   - How do we deal with this data dependency? 
 *     - For these chain that are deterministic, record the register value so that we can do functional execution of the spinlock part.
 * 5. How do we deal with spinlock?
 *   - Fast forward
 *   - Partial Functional execution on the SASS instructions
 *   - If the spinlock pattern is simple, we can replace it with PTX sim or emulation? Which is not 1-1 match of the original SASS instructions.
 *     - If we know this is just a spinlock and takes a small chunk of kernel execution, it really does not matter if we execute it faithfully or via emulation.
 *     - This way we are more accurate than fast forwarding but has less effort than exeucting SASS inst, even just a small subset of it.
 *     - TODO this requires predefining set of spinlock patterns for matching
 */

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

#define HEX(x)                                                            \
    "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x \
         << std::dec

#define CHANNEL_SIZE (1l << 20)

enum class RecvThreadState {
    WORKING,
    STOP,
    FINISHED,
};

struct CTXstate {
    /* context id */
    int id;

    /* Channel used to communicate from GPU to CPU receiving thread */
    ChannelDev* channel_dev;
    ChannelHost channel_host;

    // After initialization, set it to WORKING to make recv thread get data,
    // parent thread sets it to STOP to make recv thread stop working.
    // recv thread sets it to FINISHED when it cleans up.
    // parent thread should wait until the state becomes FINISHED to clean up.
    volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;

    // Kernel reciving state
    volatile bool kernel_receiving_done = false;

    // Kernel instruction histogram
    KernelInstructionHistogram* instr_histogram = nullptr;
};

/* lock */
pthread_mutex_t mutex;
pthread_mutex_t cuda_event_mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* grid launch id, incremented at every launch */
uint64_t global_grid_launch_id = 0;

/* Spinlock phase */
int spinlock_phase = 0;
// At end of phase 1, we will compare the two runs for each context
// to output a file containing the instructions for each kernel that are
// nondeterministic.
const int SPINLOCK_PHASE_CHECK = 1;
void* spinlock_check_thread_fun(void* args);

void* recv_thread_fun(void* args);

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_INT(spinlock_phase, "SPINLOCK_PHASE", 0, "Spinlock phase");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

    /* set mutex as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);

    pthread_mutex_init(&cuda_event_mutex, &attr);
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
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

        /* get vector of instructions of function "f" */
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);

        if (verbose) {
            printf(
                "MEMTRACE: CTX %p, Inspecting CUfunction %p name %s at address "
                "0x%lx\n",
                ctx, f, nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
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

            if (opcode_to_id_map.find(instr->getOpcode()) ==
                opcode_to_id_map.end()) {
                int opcode_id = opcode_to_id_map.size();
                opcode_to_id_map[instr->getOpcode()] = opcode_id;
                id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
            }

            int opcode_id = opcode_to_id_map[instr->getOpcode()];

            // insert instrumentation function
            nvbit_insert_call(instr, "count_instr", IPOINT_BEFORE);
            
            // pass arguments
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(instr, (uint32_t)instr->getIdx());
            nvbit_add_call_arg_launch_val64(instr, 0);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);

            cnt++;
        }
    }
}

/* flush channel */
__global__ void flush_channel(ChannelDev* ch_dev) {
    instr_count_t packet;
    // Sentinel value to indicate the end of the histogram
    packet.instr_idx = -1;
    packet.count = 0;
    ch_dev->push(&packet, sizeof(instr_count_t));
    ch_dev->flush(); 
}

void init_context_state(CUcontext ctx) {
    CTXstate* ctx_state = ctx_state_map[ctx];
    ctx_state->id = (int)ctx_state_map.size() - 1;
    ctx_state->recv_thread_done = RecvThreadState::WORKING;
    cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
    ctx_state->channel_host.init(ctx_state->id, CHANNEL_SIZE,
                                 ctx_state->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}

static void enter_kernel_launch(CUcontext ctx, CUfunction func,
                uint64_t &grid_launch_id, nvbit_api_cuda_t cbid, void* params,
                bool stream_capture = false, bool build_graph = false) {
    CTXstate* ctx_state = ctx_state_map[ctx];

    // no need to sync during stream capture or manual graph build, since no
    // kernel is actually launched.
    if (!stream_capture && !build_graph) {
        /* Make sure GPU is idle */
        cudaDeviceSynchronize();
        assert(cudaGetLastError() == cudaSuccess);
    }

    // Initialize kernel instruction histogram map
    if (ctx_state->instr_histogram == nullptr) {
        ctx_state->instr_histogram = new KernelInstructionHistogram(grid_launch_id, nvbit_get_func_name(ctx, func, true));
    } else {
        ctx_state->instr_histogram->reinit(grid_launch_id, nvbit_get_func_name(ctx, func, true));
    }

    /* instrument */
    instrument_function_if_needed(ctx, func);

    int nregs = 0;
    CUDA_SAFECALL(
        cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

    int shmem_static_nbytes = 0;
    CUDA_SAFECALL(
        cuFuncGetAttribute(&shmem_static_nbytes,
                           CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

    /* get function name and pc */
    const char* func_name = nvbit_get_func_name(ctx, func);
    uint64_t pc = nvbit_get_func_addr(ctx, func);

    // during stream capture or manual graph build, no kernel is launched, so
    // do not set launch argument, do not print kernel info, do not increase
    // grid_launch_id. All these should be done at graph node launch time.
    if (!stream_capture && !build_graph) {
        /* set grid launch id at launch time */
        nvbit_set_at_launch(ctx, func, (uint64_t)grid_launch_id);

        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            printf(
                "Spinlock: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
                "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
                "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
                "id %ld\n",
                (uint64_t)ctx, pc, func_name, grid_launch_id,
                p->config->gridDimX, p->config->gridDimY,
                p->config->gridDimZ, p->config->blockDimX,
                p->config->blockDimY, p->config->blockDimZ, nregs,
                shmem_static_nbytes + p->config->sharedMemBytes,
                (uint64_t)p->config->hStream);
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            printf(
                "Spinlock: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
                "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
                "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
                "id %ld\n",
                (uint64_t)ctx, pc, func_name, grid_launch_id, p->gridDimX,
                p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY,
                p->blockDimZ, nregs,
                shmem_static_nbytes + p->sharedMemBytes,
                (uint64_t)p->hStream);
        }

        // increment grid launch id for next launch
        // grid id can be changed here, since nvbit_set_at_launch() has copied
        // its value above.
        grid_launch_id++;
    }

    /* enable instrumented code to run */
    nvbit_enable_instrumented(ctx, func, true);

    // Reset the kernel receiving done flag for new kernel launch
    ctx_state->kernel_receiving_done = false;
}

// the function is only called for non cuda graph launch cases.
static void leave_kernel_launch(CTXstate *ctx_state, uint64_t &grid_launch_id) {
    // make sure user kernel finishes to avoid deadlock
    cudaDeviceSynchronize();
    /* push a flush channel kernel */
    flush_channel<<<1, 1>>>(ctx_state->channel_dev);

    /* Make sure GPU is idle */
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda error: %s\n", cudaGetErrorName(err));
    }
    assert(err == cudaSuccess);
    
    /* wait here until the receiving thread has not finished with the
     * current kernel */
    while (!ctx_state->kernel_receiving_done) {
        pthread_yield();
    }

    // Dump the histogram to file
    // Make a folder for the histogram
    std::string folder_name = "ctx_" + std::to_string(ctx_state->id) + "/spinlock_run_" + std::to_string(spinlock_phase);

    // Create the folder
    std::filesystem::create_directories(folder_name);

    ctx_state->instr_histogram->saveToFile( folder_name + "/" + std::to_string(ctx_state->instr_histogram->id) + "-" + ctx_state->instr_histogram->name + ".histogram");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    pthread_mutex_lock(&cuda_event_mutex);

    /* we prevent re-entry on this callback when issuing CUDA functions inside
     * this function */
    if (skip_callback_flag) {
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }
    skip_callback_flag = true;

    CTXstate* ctx_state = ctx_state_map[ctx];

    switch (cbid) {
        // Identify all the possible CUDA launch events without stream
        // parameters, they will not get involved with cuda graph
        case API_CUDA_cuLaunch:
        case API_CUDA_cuLaunchGrid:
            {
                cuLaunch_params *p = (cuLaunch_params *)params;
                CUfunction func = p->f;
                if (!is_exit) {
                    enter_kernel_launch(ctx, func, global_grid_launch_id, cbid,
                                        params);
                } else {
                    leave_kernel_launch(ctx_state, global_grid_launch_id);
                }
            } break;
        // To support kernel launched by cuda graph (in addition to existing kernel
        // launche method), we need to do:
        //
        // 1. instrument kernels at cudaGraphAddKernelNode event. This is for cases
        // that kernels are manually added to a cuda graph.
        // 2. distinguish captured kernels when kernels are recorded to a graph
        // using stream capture. cudaStreamIsCapturing() tells us whether a stream
        // is capturiong.
        // 3. per-kernel instruction counters, since cuda graph can launch multiple
        // kernels at the same time.
        //
        // Three cases:
        //
        // 1. original kernel launch:
        //     1a. for any kernel launch without using a stream, we instrument it
        //     before it is launched, call cudaDeviceSynchronize after it is
        //     launched and read the instruction counter of the kernel.
        //     1b. for any kernel launch using a stream, but the stream is not
        //     capturing, we do the same thing as 1a.
        //
        //  2. cuda graph using stream capturing: if a kernel is launched in a
        //  stream and the stream is capturing. We instrument the kernel before it
        //  is launched and do nothing after it is launched, because the kernel is
        //  not running until cudaGraphLaunch. Instead, we issue a
        //  cudaStreamSynchronize after cudaGraphLaunch is done and reset the
        //  instruction counters, since a cloned graph might be launched afterwards.
        //
        //  3. cuda graph manual: we instrument the kernel added by
        //  cudaGraphAddKernelNode and do the same thing for cudaGraphLaunch as 2.
        //
        // The above method should handle most of cuda graph launch cases.
        // kernel launches with stream parameter, they can be used for cuda graph
        case API_CUDA_cuLaunchKernel_ptsz:
        case API_CUDA_cuLaunchKernel:
        case API_CUDA_cuLaunchCooperativeKernel:
        case API_CUDA_cuLaunchCooperativeKernel_ptsz:
        case API_CUDA_cuLaunchKernelEx:
        case API_CUDA_cuLaunchKernelEx_ptsz:
        case API_CUDA_cuLaunchGridAsync:
            {
                CUfunction func;
                CUstream hStream;

                if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                    cbid == API_CUDA_cuLaunchKernelEx) {
                    cuLaunchKernelEx_params* p =
                        (cuLaunchKernelEx_params*)params;
                    func = p->f;
                    hStream = p->config->hStream;
                } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                           cbid == API_CUDA_cuLaunchKernel ||
                           cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
                           cbid == API_CUDA_cuLaunchCooperativeKernel) {
                    cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
                    func = p->f;
                    hStream = p->hStream;
                } else {
                    cuLaunchGridAsync_params* p =
                        (cuLaunchGridAsync_params*)params;
                    func = p->f;
                    hStream = p->hStream;
                }

                cudaStreamCaptureStatus streamStatus;
                /* check if the stream is capturing, if yes, do not sync */
                CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
                if (!is_exit) {
                    bool stream_capture = (streamStatus == cudaStreamCaptureStatusActive);
                    enter_kernel_launch(ctx, func, global_grid_launch_id, cbid, params, stream_capture);
                } else {
                    if (streamStatus != cudaStreamCaptureStatusActive) {
                        if (verbose >= 1) {
                            printf("kernel %s not captured by cuda graph\n", nvbit_get_func_name(ctx, func));
                        }
                        leave_kernel_launch(ctx_state, global_grid_launch_id);
                    } else {
                        if (verbose >= 1) {
                            printf("kernel %s captured by cuda graph\n", nvbit_get_func_name(ctx, func));
                        }
                    }
                }
            } break;
        case API_CUDA_cuGraphAddKernelNode:
            {
                cuGraphAddKernelNode_params *p = (cuGraphAddKernelNode_params *)params;
                CUfunction func = p->nodeParams->func;

                if (!is_exit) {
                    // cuGraphAddKernelNode_params->nodeParams is the same as
                    // cuLaunchKernel_params up to sharedMemBytes
                    enter_kernel_launch(ctx, func, global_grid_launch_id, cbid, (void*)p->nodeParams, false, true);
                } 
            } break;
        case API_CUDA_cuGraphLaunch:
            {
                // if we are exiting a cuda graph launch:
                // Wait until the graph is completed using
                // cudaStreamSynchronize()
                if (is_exit) {
                    cuGraphLaunch_params *p = (cuGraphLaunch_params *)params;

                    CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
                    assert(cudaGetLastError() == cudaSuccess);
                    /* push a flush channel kernel */
                    flush_channel<<<1, 1, 0, p->hStream>>>(ctx_state->channel_dev);
                    CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
                    assert(cudaGetLastError() == cudaSuccess);
                }

            } break;
        default:
            break;
    };


    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;

    pthread_mutex_lock(&mutex);
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    ChannelHost* ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);

    while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        /* receive buffer from channel */
        uint32_t num_recv_bytes = 0;
        if (!ctx_state->kernel_receiving_done &&
            (num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE)) > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                instr_count_t* packet =
                    (instr_count_t*)&recv_buffer[num_processed_bytes];
                
                // Sentinel value to indicate the end of the histogram
                if (packet->instr_idx == -1) {
                    ctx_state->kernel_receiving_done = true;
                    break;
                }

                // Increment the instruction count
                ctx_state->instr_histogram->add(packet->instr_idx, packet->count);
                if (verbose >= 1) {
                    printf("Spinlock: Incrementing instruction count for instruction %d by %d\n", packet->instr_idx, packet->count);
                }

                num_processed_bytes += sizeof(instr_count_t);
            }
        }
    }
    free(recv_buffer);
    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    if (verbose) {
        printf("Spinlock: STARTING CONTEXT %p\n", ctx);
    }
    assert(ctx_state_map.find(ctx) == ctx_state_map.end());
    CTXstate* ctx_state = new CTXstate;
    ctx_state_map[ctx] = ctx_state;
    pthread_mutex_unlock(&mutex);
}

void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    init_context_state(ctx);
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;
    if (verbose) {
        printf("Spinlock: TERMINATING CONTEXT %p\n", ctx);
    }
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Notify receiver thread and wait for receiver thread to
     * notify back */
    ctx_state->recv_thread_done = RecvThreadState::STOP;
    while (ctx_state->recv_thread_done != RecvThreadState::FINISHED)
        ;

    // Check for spinlock
    pthread_t spinlock_check_thread;
    if (spinlock_phase == SPINLOCK_PHASE_CHECK) {
        // Compare the two runs for each context
        // to output a file containing the instructions for each kernel that are
        // nondeterministic.
        // Spawn a thread to identify the spinlock sections of
        // kernels in this context
        printf("Spinlock: Spawning thread to check for spinlock in context %d\n", ctx_state->id);
        pthread_create(&spinlock_check_thread, NULL, spinlock_check_thread_fun, (void*)ctx_state->id);
    }
    // Clean up
    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;
    // Wait for the spinlock check thread to finish
    if (spinlock_phase == SPINLOCK_PHASE_CHECK) {
        pthread_join(spinlock_check_thread, NULL);
    }
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_graph_node_launch(CUcontext ctx, CUfunction func,
                                          CUstream stream,
                                          uint64_t launch_handle) {
    func_config_t config = {0};
    const char* func_name = nvbit_get_func_name(ctx, func);
    uint64_t pc = nvbit_get_func_addr(ctx, func);

    pthread_mutex_lock(&mutex);
    nvbit_set_at_launch(ctx, func, (uint64_t)global_grid_launch_id, stream,
                        launch_handle);
    nvbit_get_func_config(ctx, func, &config);

    printf(
        "Spinlock: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
        "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
        "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
        "id %ld\n",
        (uint64_t)ctx, pc, func_name, global_grid_launch_id, config.gridDimX,
        config.gridDimY, config.gridDimZ, config.blockDimX, config.blockDimY,
        config.blockDimZ, config.num_registers,
        config.shmem_static_nbytes + config.shmem_dynamic_nbytes,
        (uint64_t)stream);
    // grid id can be changed here, since nvbit_set_at_launch() has copied its
    // value above.
    global_grid_launch_id++;
    pthread_mutex_unlock(&mutex);
}

void* spinlock_check_thread_fun(void* args) {
    uint32_t ctx_id = (uint64_t)args;
    std::string context_folder = "ctx_" + std::to_string(ctx_id);
    
    // Scan the context folder for spinlock_run_* folders
    std::string spinlock_run0_folder = context_folder + "/spinlock_run_0";
    std::string spinlock_run1_folder = context_folder + "/spinlock_run_1";

    // Get the list of histogram files in each spinlock run folder and serialize
    std::vector<KernelInstructionHistogram*> spinlock_run0_histograms;
    std::vector<KernelInstructionHistogram*> spinlock_run1_histograms;

    // Load the histogram files
    printf("Spinlock: Loading histograms from %s and %s\n", spinlock_run0_folder.c_str(), spinlock_run1_folder.c_str());
    std::map<std::string, std::vector<KernelInstructionHistogram*>*> zipped_folders = {{spinlock_run0_folder, &spinlock_run0_histograms}, {spinlock_run1_folder, &spinlock_run1_histograms}};
    for (const auto& iter : zipped_folders) {
        auto spinlock_run_folder = iter.first;
        auto histograms = iter.second;
        printf("Spinlock: Loading histograms from %s\n", spinlock_run_folder.c_str());
        for (const auto& entry : std::filesystem::directory_iterator(spinlock_run_folder)) {
            if (entry.is_regular_file() && entry.path().extension().compare(".histogram") == 0) {
                printf("Spinlock: Loading histogram from %s\n", entry.path().string().c_str());
                KernelInstructionHistogram *histogram = new KernelInstructionHistogram();
                histogram->loadFromFile(entry.path().string());
                histograms->push_back(histogram);
            }
        }
    }
    printf("Spinlock: Loaded %d histograms from %s and %d histograms from %s\n", spinlock_run0_histograms.size(), spinlock_run0_folder.c_str(), spinlock_run1_histograms.size(), spinlock_run1_folder.c_str());
    // Check if the kernel count are the same
    assert(spinlock_run0_histograms.size() == spinlock_run1_histograms.size());

    printf("Spinlock: Comparing the two histograms\n");
    // Now compare the two histograms and generate output of spinlock instructions per context
    // Each row will be kernel id, kernel name, and indices of spinlock instructions
    std::string output_file = context_folder + "/spinlock_instructions.txt";
    std::ofstream output_file_stream(output_file);
    printf("Spinlock: Generating output file %s\n", output_file.c_str());
    for (auto run0_histogram : spinlock_run0_histograms) {
        printf("Spinlock: Comparing histogram %d %s\n", run0_histogram->id, run0_histogram->name.c_str());
        auto run1_histogram = spinlock_run1_histograms.at(run0_histogram->id);
        auto spinlock_instructions = run0_histogram->findSpinlock(*run1_histogram);
        printf("Spinlock: Found %d spinlock instructions\n", spinlock_instructions.size());
        output_file_stream << run0_histogram->id << ", " << run0_histogram->name << ": ";
        for (auto [instr_idx, counts] : spinlock_instructions) {
            // Write to output file
            output_file_stream << instr_idx << ", ";
        }
        output_file_stream << "\n";
    }
    output_file_stream.close();
    printf("Spinlock: Generated output file %s\n", output_file.c_str());
    // Free the histograms
    for (auto histogram : spinlock_run0_histograms) {
        delete histogram;
    }
    for (auto histogram : spinlock_run1_histograms) {
        delete histogram;
    }
    return NULL;
}
