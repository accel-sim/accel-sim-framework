/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_set>
#include <iostream>
#include <map>
#include <fstream>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_instrs = 0;

/* counter to memcpy calls */
uint64_t tot_memcpy_h2d = 0;
uint64_t tot_memcpy_d2h = 0;

/* Trace file pointer */
FILE * traceFp;

/* Hashmap to keep device pointer reference */
std::map<CUdeviceptr, char*> *dptr_map;

/* kernel instruction counter, updated by the GPU */
__managed__ uint64_t counter = 0;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t start_grid_num = 0;
uint32_t end_grid_num = UINT32_MAX;
int verbose = 0;
int count_warp_level = 1;
int exclude_pred_off = 0;
int active_from_start = 1;
bool mangled = false;

/* used to select region of insterest when active from start is off */
bool active_region = true;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(start_grid_num, "START_GRID_NUM", 0,
                "Beginning of the kernel gird launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(
        end_grid_num, "END_GRID_NUM", UINT32_MAX,
        "End of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
                "Count warp level or thread level instructions");
    GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 0,
                "Exclude predicated off instruction from count");
    GET_VAR_INT(
        active_from_start, "ACTIVE_FROM_START", 1,
        "Start instruction counting from start or wait for cuProfilerStart "
        "and cuProfilerStop");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1,
                "Print kernel names mangled or not");

    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    if (active_from_start == 0) {
        active_region = false;
    }

    // Init trace fp
    traceFp = fopen("cuda_calls.trace", "w");

    // Init map
    dptr_map = new std::map<CUdeviceptr, char*>();

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
// TODO: Balar runs on api calls, will this cause issues?
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (is_exit) {
        // Check memalloc result after it finishes
        // Deref as the argument is passed by addr
        if (cbid == API_CUDA_cuMemAlloc || cbid == API_CUDA_cuMemAlloc_v2) {
            // TODO Use address instead of value to keep reference?
            cuMemAlloc_v2_params *p = (cuMemAlloc_v2_params *)params;
            uint64_t size = dptr_map->size();

            // Assuming there will not be 10^24 device pointer exist
            char *name = (char*)malloc(30);
            sprintf(name, "dptr-%d", size);
            dptr_map->insert({*(p->dptr), name});

            fprintf(traceFp, "CUDA memalloc: dptr: %s, size: %d\n", name, p->bytesize);
        } else if (cbid == API_CUDA_cuMemcpyDtoH
            || cbid == API_CUDA_cuMemcpyDtoHAsync
            || cbid == API_CUDA_cuMemcpyDtoH_v2
            || cbid == API_CUDA_cuMemcpyDtoHAsync_v2
            || cbid == API_CUDA_cuMemcpyDtoH_v2_ptds
            || cbid == API_CUDA_cuMemcpyDtoHAsync_v2_ptsz) {
            // Move to here as we need to save the data for verification purpose
            cuMemcpyDtoH_v2_params *p = (cuMemcpyDtoH_v2_params *) params;
            CUdeviceptr dptr = p->srcDevice;
            char* name = dptr_map->find(dptr)->second;
            
            // Dump src data (or dst here as we finished copy) to a file with name: cuMemcpyD2H-COUNT-SIZE.data
            char buf[200];
            sprintf(buf, "cuMemcpyD2H-%d-%d.data", tot_memcpy_d2h, p->ByteCount);
            std::string filename(buf);
            std::ofstream out(filename);
            if (out.is_open()) {
                out.write((const char *)p->dstHost, p->ByteCount);
                out.close();
            } else {
                std::cerr << "Cannot open " << filename << std::endl;
            }
            
            tot_memcpy_d2h++;

            fprintf(traceFp, "CUDA memcpyD2H detected: host_ptr: %p, device_ptr: %s, size: %ld, data_file: %s\n", p->dstHost, name, p->ByteCount, buf);
        }
        return;
    }

    if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {
        /* cast params to cuLaunchKernel_params since if we are here we know these are
         * the right parameters type */
        
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
        std::string funcName(nvbit_get_func_name(ctx, p->f, false));
        std::string funcNamePTX(nvbit_get_func_name(ctx, p->f, true));
        std::size_t start = funcName.find("(");
        std::size_t end = funcName.find(")");
        std::string funcTypes = funcName.substr(start + 1, end - start - 1);

        fprintf(traceFp, "CUDA cu kernel launch detected: name: %s, ptx_name: %s, "
               "funcptr: %p, gdx: %d, gdy: %d, gdz: %d, "
               "bdx: %d, bdy: %d, bdz: %d, sharedBytes: %d, "
               "CUstream: %p, args: ", 
               funcName.substr(0, start).c_str(), 
               funcNamePTX.c_str(),
               p->f, p->gridDimX, p->gridDimY, p->gridDimZ,
               p->blockDimX, p->blockDimY, p->blockDimZ, 
               p->sharedMemBytes, p->hStream);
        
        // TODO: Need to get kernel param count and size and offset information
        // Can use nvbit_get_kernel_argument_sizes (nvbit 1.5.5 has problem with this)
        // Can also use nvbit_get_func_name
        // Currently parse the function signature instead, 
        // will have some compactibility issues
        void** tmp = (p->kernelParams);

        // Parse each argument type
        while (funcTypes.size() > 0) {
            std::string type;
            std::string delim(", ");
            std::size_t firstSplit = funcTypes.find(delim);
            if (firstSplit == std::string::npos) {
                // Reach last argument
                type = funcTypes;
                funcTypes.clear();
            } else {
                type = funcTypes.substr(0, firstSplit);
                funcTypes = funcTypes.substr(firstSplit + delim.length());
            }

            // Parse on type and give type size
            if (type.compare("double*") == 0) {
                CUdeviceptr dptr = (CUdeviceptr) *(double **)(*tmp);
                char* name = dptr_map->find(dptr)->second;
                fprintf(traceFp, "%s/%d/", name, sizeof(double*));
            } else if (type.compare("float*") == 0) {
                CUdeviceptr dptr = (CUdeviceptr) *(float **)(*tmp);
                char* name = dptr_map->find(dptr)->second;
                fprintf(traceFp, "%s/%d/", name, sizeof(float*));
            } else if (type.compare("int*") == 0) {
                CUdeviceptr dptr = (CUdeviceptr) *(int **)(*tmp);
                char* name = dptr_map->find(dptr)->second;
                fprintf(traceFp, "%s/%d/", name, sizeof(int*));
            } else if (type.compare("double") == 0) {
                fprintf(traceFp, "%f/%d/", *(double *)(*tmp), sizeof(double));
            } else if (type.compare("float") == 0) {
                fprintf(traceFp, "%f/%d/", *(float *)(*tmp), sizeof(float));
            } else if (type.compare("int") == 0) {
                fprintf(traceFp, "%d/%d/", *(int *)(*tmp), sizeof(int));
            }

            // Increment the argument pointer
            tmp++;
        }


        // TODO: Cannot use this until next issue
        // TODO: See: https://github.com/NVlabs/NVBit/issues/80
        /**
        std::vector<int> sizes = nvbit_get_kernel_argument_sizes(p->f);
        void** tmp = (p->kernelParams);

        for (auto it = sizes.begin(); it < sizes.end(); it++) {
            int size = *it;
            if (size == 1 << 0) {
                printf("%d, ", *(uint8_t *)(*tmp));
            } else if (size == 1 << 1) {
                printf("%d, ", *(uint16_t *)(*tmp));
            } else if (size == 1 << 2) {
                printf("%d, ", *(uint32_t *)(*tmp));
            } else if (size == 1 << 3) {
                printf("%p, ", *(uint64_t *)(*tmp));
            } else {
                printf("Unknown type, ");
            }
            tmp++;
        }*/

        // // TODO: Hard coded for testing
        // printf("%p, ", *((double **)(*tmp)));
        // tmp++;
        // printf("%p, ", *((double **)(*tmp)));
        // tmp++;
        // printf("%p, ", *((double **)(*tmp)));
        // tmp++;
        // printf("%d", *((int *)(*tmp)));

        fprintf(traceFp, "\n");

    } else if (cbid == API_CUDA_cuMemcpyHtoD 
            || cbid == API_CUDA_cu64MemcpyHtoD 
            || cbid == API_CUDA_cuMemcpyHtoDAsync
            || cbid == API_CUDA_cu64MemcpyHtoDAsync
            || cbid == API_CUDA_cuMemcpyHtoD_v2
            || cbid == API_CUDA_cuMemcpyHtoDAsync_v2
            || cbid == API_CUDA_cuMemcpyHtoD_v2_ptds
            || cbid == API_CUDA_cuMemcpyHtoDAsync_v2_ptsz) {
        cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *) params;
        CUdeviceptr dptr = p->dstDevice;
        char* name = dptr_map->find(dptr)->second;
        
        // Dump src data to a file with name: cuMemcpyH2D-COUNT-SIZE.data
        char buf[200];
        sprintf(buf, "cuMemcpyH2D-%d-%d.data", tot_memcpy_h2d, p->ByteCount);
        std::string filename(buf);
        std::ofstream out(filename);
        if (out.is_open()) {
            out.write((const char *)p->srcHost, p->ByteCount);
            out.close();
        } else {
            std::cerr << "Cannot open " << filename << std::endl;
        }
        
        tot_memcpy_h2d++;

        fprintf(traceFp, "CUDA memcpyH2D detected: device_ptr: %s, host_ptr: %p, size: %d, data_file: %s\n", name, p->srcHost, p->ByteCount, buf);
    } else if (cbid == API_CUDA_cuMemFree || cbid == API_CUDA_cuMemFree_v2) {
        cuMemFree_v2_params *p = (cuMemFree_v2_params *) params;
        CUdeviceptr dptr = p->dptr;
        char* name = dptr_map->find(dptr)->second;
        fprintf(traceFp, "CUDA free detected: dptr: %s\n", name);
    }
}

void nvbit_at_term() {
    printf("Finishes cuda api call tracing\n");
    fclose(traceFp);
}
