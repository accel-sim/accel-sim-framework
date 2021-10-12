#!/usr/bin/python3

# Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
# Northwestern University, Purdue University, The University of British Columbia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer;
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution;
# 3. Neither the names of Northwestern University, Purdue University,
#    The University of British Columbia nor the names of their contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import pandas as pd
import numpy as np
import os
import sys
import re
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-d", "--dir", dest="hw_stats_dir",
                 help="directory for all hw stats. Usally under <./hw_run/device-#>")
# parser.add_option("-v", "--accelwattch_validation", dest="validation", action="store_true",
#                  help="")

(options, args) = parser.parse_args()
hw_stats_dir = options.hw_stats_dir
path = []
start_line = []
for dirpath, dirnames, filenames in os.walk(hw_stats_dir):
    for file in filenames:
        if "csv.nsight" in file:
            path.append(os.path.join(dirpath, file))
            # print (os.path.join(dirpath, file))

kernel_name_map = {
    # nsight name : accel-sim name
    "findRangeK" : "findRangeK",
    "findK": "findK",
    "bpnn_layerforward_CUDAfloatfloatfloatfloatintint" : "_Z22bpnn_layerforward_CUDAPfS_S_S_ii",
    "bpnn_adjust_weights_cudafloatintfloatintfloatfloat" : "_Z24bpnn_adjust_weights_cudaPfiS_iS_S_",
    "binomialOptionsKernel" : "_Z21binomialOptionsKernelv",
    "CUDAkernel1DCTfloatintintint" : "_Z14CUDAkernel1DCTPfiiiy",
    "CUDAkernel2DCTfloatfloatint": "_Z14CUDAkernel2DCTPfS_i", 
    "fwtBatch2Kernelfloatfloatint" : "_Z15fwtBatch2KernelPfS_i",
    "fwtBatch1Kernelfloatfloatint" : "_Z15fwtBatch1KernelPfS_i",
    "histogram64Kernelunsignedintuint4unsignedint" : "_Z17histogram64KernelPjP5uint4j",
    "calculate_tempintfloatfloatfloatintintintintfloatfloatfloatfloatfloatfloat" : "_Z14calculate_tempiPfS_S_iiiiffffff",
    "kmeansPointfloatintintintintfloatfloatint" : "_Z11kmeansPointPfiiiPiS_S_S0_",
    "voidmergeElementaryIntervalsKernel1uunsignedintunsignedintunsignedintunsignedintunsignedintunsignedintunsignedintunsignedint" : "_Z30mergeElementaryIntervalsKernelILj1EEvPjS0_S0_S0_S0_S0_jj",
    "voidmergeSortSharedKernel1uunsignedintunsignedintunsignedintunsignedintunsignedint" : "_Z21mergeSortSharedKernelILj1EEvPjS0_S0_S0_j",
    "ComputeQ_GPUintintfloatfloatfloatfloatfloat" : "_Z12ComputeQ_GPUiiPfS_S_S_S_",
    "mb_sad_calcunsignedshortunsignedshortintint" : "_Z11mb_sad_calcPtS_ii",
    "mysgemmNTfloatconstintfloatconstintfloatintintfloatfloat" : "_Z9mysgemmNTPKfiS0_iPfiiff",
    "quasirandomGeneratorKernelfloatunsignedintunsignedint" : "_Z26quasirandomGeneratorKernelPfjj",
    "inverseCNDKernelfloatunsignedintunsignedint" : "_Z16inverseCNDKernelPfPjj",
    "sobolGPU_kernelunsignedintunsignedintunsignedintfloat" : "_Z15sobolGPU_kerneljjPjPf",
    "sradfloatintintlongintintintintfloatfloatfloatfloatfloatfloatfloat" : "_Z4sradfiilPiS_S_S_PfS0_S0_S0_fS0_S0_",
    "_ZN7cutlass4gemm16gemm_kernel_nolbINS0_12GemmMainloopINS0_10GemmTraitsINS0_14WmmaGemmConfigILNS_12MatrixLayout4KindE1ELS6_1ENS_5ShapeILi64ELi128ELi128ELi1EEE6__halfS9_ffNS7_ILi64ELi32ELi64ELi1EEENS7_ILi16ELi16ELi16ELi1EEELi8ELi8ELi8ELi8ELi4ELi4ELi4EEENS0_16GlobalLoadStreamILNS_11GemmOperand4KindE0ENS0_20GemmGlobalIteratorAbINS0_20GemmGlobalTileTraitsILSF_0ELS6_1EKS9_NS7_ILi1ELi64ELi128ELi1EEENS7_ILi1ELi8ELi32ELi1EEELi8EEEiEENS_17TileStoreIteratorINS0_27GemmSharedStoreTileAbTraitsIS9_NS7_ILi1ELi64ELi136ELi1EEENS7_ILi1ELi16ELi16ELi1EEELi8EEES9_LNS_15IteratorAdvance4KindE1ELNS_11MemorySpace4KindE1EiS9_LNS_19FragmentElementType4KindE0ENS7_ILi0ELi0ELi0ELi0EEEEENS_4CopyINS_8FragmentIS9_Li32ELm16EEEEEEENSD_ILSF_1ENSG_INSH_ILSF_1ELS6_1ESI_NS7_ILi1ELi128ELi64ELi1EEENS7_ILi1ELi4ELi64ELi1EEELi8EEEiEENSN_INSO_IS9_NS7_ILi1ELi128ELi72ELi1EEENS7_ILi1ELi32ELi8ELi1EEELi8EEES9_LST_1ELSV_1EiS9_LSX_0ESY_EES13_EENS0_16SharedLoadStreamINS_16TileLoadIteratorINS0_29WmmaGemmSharedLoadTileATraitsILS6_1ES9_SP_NS7_ILi1ELi4ELi2ELi1EEELi16ENS7_ILi1ELi1ELi4ELi1EEENS7_ILi2176ELi0ELi32ELi0EEESB_EES9_LST_1ELSV_1EiNS_10WmmaMatrixILSF_0ELS6_1ES9_SB_EELSX_1ESY_EENS10_INS11_IS1M_Li4ELm16EEEEEEENS1E_INS1F_INS0_29WmmaGemmSharedLoadTileBTraitsILS6_1ES9_S19_S1H_Li1152ENS7_ILi1ELi1ELi2ELi1EEENS7_ILi16ELi0ELi4608ELi1EEESB_EES9_LST_1ELSV_1EiNS1L_ILSF_1ELS6_1ES9_SB_EELSX_1ESY_EENS10_INS11_IS1V_Li2ELm16EEEEEEENS0_12GemmEpilogueINS0_28SimplifiedGemmEpilogueTraitsISC_NS0_13LinearScalingIfNS0_19FragmentMultiplyAddIffLb1EEEEEiNS0_28WmmaGemmEpilogueTraitsHelperISC_fS25_iEEEEEENS0_20IdentityBlockSwizzleEiNS0_17ClearAccumulatorsIfLi1EEEEEEEEEvNT_6ParamsE" : "_ZN7cutlass4gemm16gemm_kernel_nolbINS0_12GemmMainloopINS0_10GemmTraitsINS0_14WmmaGemmConfigILNS_12MatrixLayout4KindE1ELS6_1ENS_5ShapeILi64ELi128ELi128ELi1EEE6__halfS9_ffNS7_ILi64ELi32ELi64ELi1EEENS7_ILi16ELi16ELi16ELi1EEELi8ELi8ELi8ELi8ELi4ELi4ELi4EEENS0_16GlobalLoadStreamILNS_11GemmOperand4KindE0ENS0_20GemmGlobalIteratorAbINS0_20GemmGlobalTileTraitsILSF_0ELS6_1EKS9_NS7_ILi1ELi64ELi128ELi1EEENS7_ILi1ELi8ELi32ELi1EEELi8EEEiEENS_17TileStoreIteratorINS0_27GemmSharedStoreTileAbTraitsIS9_NS7_ILi1ELi64ELi136ELi1EEENS7_ILi1ELi16ELi16ELi1EEELi8EEES9_LNS_15IteratorAdvance4KindE1ELNS_11MemorySpace4KindE1EiS9_LNS_19FragmentElementType4KindE0ENS7_ILi0ELi0ELi0ELi0EEEEENS_4CopyINS_8FragmentIS9_Li32ELm16EEEEEEENSD_ILSF_1ENSG_INSH_ILSF_1ELS6_1ESI_NS7_ILi1ELi128ELi64ELi1EEENS7_ILi1ELi4ELi64ELi1EEELi8EEEiEENSN_INSO_IS9_NS7_ILi1ELi128ELi72ELi1EEENS7_ILi1ELi32ELi8ELi1EEELi8EEES9_LST_1ELSV_1EiS9_LSX_0ESY_EES13_EENS0_16SharedLoadStreamINS_16TileLoadIteratorINS0_29WmmaGemmSharedLoadTileATraitsILS6_1ES9_SP_NS7_ILi1ELi4ELi2ELi1EEELi16ENS7_ILi1ELi1ELi4ELi1EEENS7_ILi2176ELi0ELi32ELi0EEESB_EES9_LST_1ELSV_1EiNS_10WmmaMatrixILSF_0ELS6_1ES9_SB_EELSX_1ESY_EENS10_INS11_IS1M_Li4ELm16EEEEEEENS1E_INS1F_INS0_29WmmaGemmSharedLoadTileBTraitsILS6_1ES9_S19_S1H_Li1152ENS7_ILi1ELi1ELi2ELi1EEENS7_ILi16ELi0ELi4608ELi1EEESB_EES9_LST_1ELSV_1EiNS1L_ILSF_1ELS6_1ES9_SB_EELSX_1ESY_EENS10_INS11_IS1V_Li2ELm16EEEEEEENS0_12GemmEpilogueINS0_28SimplifiedGemmEpilogueTraitsISC_NS0_13LinearScalingIfNS0_19FragmentMultiplyAddIffLb1EEEEEiNS0_28WmmaGemmEpilogueTraitsHelperISC_fS25_iEEEEEENS0_20IdentityBlockSwizzleEiNS0_17ClearAccumulatorsIfLi1EEEEEEEEEvNT_6ParamsE",
    "compute_gemm__halfconst__halfconstfloatconstfloatfloatfloat" : "_Z12compute_gemmPK6__halfS1_PKfPfff"
}

# increase this number to match total number of metrics
size = 41

total = np.zeros([1,size])
for file in path:
    print(file)
    # skip first several lines of nsight output
    with open(file) as myFile:
        for num, line in enumerate(myFile, 1):
            if "Disconnected" in line:
                start = num
                break
    if start == 0:
        print("output file error", file)
        break
    df = pd.read_csv(file,',',skiprows = start)

    # add metrics here
    app_names = df["Process Name"].to_numpy()[1:].astype(str)
    kernel_names = df["Kernel Name"].to_numpy()[1:].astype(str)
    sm__inst_executed = df["sm__inst_executed.sum"].to_numpy()[1:].astype(int)
    sm__sass_thread_inst_executed_op_fp16_pred_on = df["sm__sass_thread_inst_executed_op_fp16_pred_on.sum"].to_numpy()[1:].astype(int)
    sm__sass_thread_inst_executed_op_fp32_pred_on = df["sm__sass_thread_inst_executed_op_fp32_pred_on.sum"].to_numpy()[1:].astype(int)
    sm__sass_thread_inst_executed_op_fp64_pred_on = df["sm__sass_thread_inst_executed_op_fp64_pred_on.sum"].to_numpy()[1:].astype(int)
    sm__sass_thread_inst_executed_op_integer_pred_on = df["sm__sass_thread_inst_executed_op_integer_pred_on.sum"].to_numpy()[1:].astype(int)
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit = df["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"].to_numpy()[1:].astype(int)
    sm__sass_thread_inst_executed_op_dfma_pred_on = df["sm__sass_thread_inst_executed_op_dfma_pred_on.sum"].to_numpy()[1:].astype(int)
    sm__sass_thread_inst_executed_op_dmul_pred_on = df["sm__sass_thread_inst_executed_op_dmul_pred_on.sum"].to_numpy()[1:].astype(int)
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss = df["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"].to_numpy()[1:].astype(int)
    l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit = df["l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum"].to_numpy()[1:].astype(int)
    l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss = df["l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum"].to_numpy()[1:].astype(int)
    idc__requests = df["idc__requests.sum"].to_numpy()[1:].astype(int)
    idc__requests_hit = df["idc__requests_lookup_hit.sum"].to_numpy()[1:].astype(int)
    sm__sass_inst_executed_op_shared_ld = df["sm__sass_inst_executed_op_shared_ld.sum"].to_numpy()[1:].astype(int)
    sm__sass_inst_executed_op_shared_st = df["sm__sass_inst_executed_op_shared_st.sum"].to_numpy()[1:].astype(int)
    sm__inst_executed_pipe_alu = df["sm__inst_executed_pipe_alu.sum"].to_numpy()[1:].astype(int)
    sm__inst_executed_pipe_fma = df["sm__inst_executed_pipe_fma.sum"].to_numpy()[1:].astype(int)
    sm__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on = df["sm__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on.sum"].to_numpy()[1:].astype(int)
    sm__inst_executed_pipe_xu = df["sm__inst_executed_pipe_xu.sum"].to_numpy()[1:].astype(int)
    sm__inst_executed_pipe_tensor = df["sm__inst_executed_pipe_tensor.sum"].to_numpy()[1:].astype(int)
    sm__inst_executed_pipe_tex = df["sm__inst_executed_pipe_tex.sum"].to_numpy()[1:].astype(int)
    sm__inst_executed_pipe_fp16 = df["sm__inst_executed_pipe_fp16.sum"].to_numpy()[1:].astype(int)
    sm__inst_executed_pipe_fp64 = df["sm__inst_executed_pipe_fp64.sum"].to_numpy()[1:].astype(int)
    sm__pipe_alu_cycles_active = df["sm__pipe_alu_cycles_active.sum"].to_numpy()[1:].astype(int)
    sm__pipe_fma_cycles_active = df["sm__pipe_fma_cycles_active.sum"].to_numpy()[1:].astype(int)
    sm__pipe_fp64_cycles_active = df["sm__pipe_fp64_cycles_active.sum"].to_numpy()[1:].astype(int)
    sm__pipe_shared_cycles_active = df["sm__pipe_shared_cycles_active.sum"].to_numpy()[1:].astype(int)
    sm__pipe_tensor_cycles_active = df["sm__pipe_tensor_cycles_active.sum"].to_numpy()[1:].astype(int)
    lts__t_sectors_srcunit_tex_op_read_lookup_hit = df["lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"].to_numpy()[1:].astype(int)
    lts__t_sectors_srcunit_tex_op_read_lookup_miss = df["lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"].to_numpy()[1:].astype(int)
    lts__t_sectors_srcunit_tex_op_write_lookup_hit = df["lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum"].to_numpy()[1:].astype(int)
    lts__t_sectors_srcunit_tex_op_write_lookup_miss = df["lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum"].to_numpy()[1:].astype(int)
    lts__t_sectors_srcunit_tex_op_read = df["lts__t_sectors_srcunit_tex_op_read.sum"].to_numpy()[1:].astype(int)
    lts__t_sectors_srcunit_tex_op_write = df["lts__t_sectors_srcunit_tex_op_write.sum"].to_numpy()[1:].astype(int)
    dram__sectors_read = df["dram__sectors_read.sum"].to_numpy()[1:].astype(int)
    dram__sectors_write = df["dram__sectors_write.sum"].to_numpy()[1:].astype(int)
    sm__cycles_active = df["sm__cycles_active.avg"].to_numpy()[1:].astype(float)
    sm__cycles_active_total = df["sm__cycles_active.sum"].to_numpy()[1:].astype(float)
    sm__cycles_elapsed_avg = df["sm__cycles_elapsed.avg"].to_numpy()[1:].astype(float)
    sm__cycles_elapsed_total = df["sm__cycles_elapsed.sum"].to_numpy()[1:].astype(int)
    sm__warps_active = df["sm__warps_active.avg.pct_of_peak_sustained_active"].to_numpy()[1:].astype(float)
    active_threads = sm__warps_active * 32 / 100
    smsp__cycles_active = df["smsp__cycles_active.avg.pct_of_peak_sustained_elapsed"].to_numpy()[1:].astype(float)
    smsp__thread_inst_executed_per_inst_executed = df["smsp__thread_inst_executed_per_inst_executed.ratio"].to_numpy()[1:].astype(float)
    sm__sass_inst_executed_op_memory_128b = df["sm__sass_inst_executed_op_memory_128b.sum"].to_numpy()[1:].astype(float)
    sm__sass_inst_executed_op_memory_64b = df["sm__sass_inst_executed_op_memory_64b.sum"].to_numpy()[1:].astype(float)
    sm__sass_inst_executed_op_memory_32b = df["sm__sass_inst_executed_op_memory_32b.sum"].to_numpy()[1:].astype(float)
    sm__sass_inst_executed_op_memory_16b = df["sm__sass_inst_executed_op_memory_16b.sum"].to_numpy()[1:].astype(float)
    sm__sass_inst_executed_op_memory_8b = df["sm__sass_inst_executed_op_memory_8b.sum"].to_numpy()[1:].astype(float)
    sm__sass_thread_inst_executed = df["sm__sass_thread_inst_executed.sum"].to_numpy()[1:].astype(int)


    zeros = np.zeros(kernel_names.shape).astype(int)

    # start processing data
    # Please don't append metrics in the middle
    # Some metrics processing are depending on constant column number
    # Adding metrics to the end of the metrics should be fine.
    # Remember to change the <size> at line 25
    # +1 for each column (each new metric)

    kernel_output = []
    # kernel name, 0
    kernel_output.append(kernel_names)
    # insts, 1
    kernel_output.append(sm__inst_executed)
    # fp insts + int insts, 2
    fp_int_inst = sm__inst_executed - sm__sass_inst_executed_op_memory_128b - sm__sass_inst_executed_op_memory_64b - sm__sass_inst_executed_op_memory_32b - sm__sass_inst_executed_op_memory_16b - sm__sass_inst_executed_op_memory_8b
    kernel_output.append(fp_int_inst)
    # inst cache, 3
    kernel_output.append(zeros)
    # 4
    kernel_output.append(zeros)

    # L1 Read Hits, 5
    kernel_output.append(l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit)
    # L1 Read Miss, 6
    kernel_output.append(l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss)
    # L1 write Hit, 7
    kernel_output.append(l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit)
    # L1 write Hit, 8
    kernel_output.append(l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss)
    # constant cache, 9
    kernel_output.append(idc__requests)
    # shared memory, 10
    kernel_output.append(sm__sass_inst_executed_op_shared_ld + sm__sass_inst_executed_op_shared_st)
    # register, 11
    kernel_output.append(zeros)
    # 12
    kernel_output.append(zeros)
    # 13
    kernel_output.append(zeros)
    # total ALU, 14
    alu = sm__inst_executed_pipe_alu
    kernel_output.append(alu)
    # 15
    kernel_output.append(zeros)
    # 16
    kernel_output.append(zeros)
    # int and fp mul/mad, 17
    fma = sm__inst_executed_pipe_fma
    kernel_output.append(fma)
    # 18
    kernel_output.append(zeros)
    xu = sm__inst_executed_pipe_xu
    # 19
    kernel_output.append(xu)
    # 20
    kernel_output.append(zeros)
    # 21
    kernel_output.append(zeros)
    # 22
    kernel_output.append(zeros)
    # Double Precision Mul/Mad thread insts, 23
    kernel_output.append(sm__sass_thread_inst_executed_op_dfma_pred_on + sm__sass_thread_inst_executed_op_dmul_pred_on)
    # Tensor Core thread insts, 24
    tensor = sm__inst_executed_pipe_tensor
    kernel_output.append(tensor)
    # Texture Unit accesses, 25
    tex = sm__inst_executed_pipe_tex
    kernel_output.append(tex)
    # DRAM reads, 26
    kernel_output.append(dram__sectors_read)
    # DRAM writes, 27
    kernel_output.append(dram__sectors_write)
    # DRAM precharge, 28
    kernel_output.append(zeros)
    # L2 read Hits, 29
    kernel_output.append(lts__t_sectors_srcunit_tex_op_read_lookup_hit)
    # L2 read Miss, 30
    kernel_output.append(lts__t_sectors_srcunit_tex_op_read_lookup_miss)
    # L2 Write Hits, 31
    kernel_output.append(lts__t_sectors_srcunit_tex_op_write_lookup_hit)
    # L2 Write Miss, 32
    kernel_output.append(lts__t_sectors_srcunit_tex_op_write_lookup_miss)
    # Flits from SMs to memory partitions + Flits from memory partitions to SMs, 33
    flits = (lts__t_sectors_srcunit_tex_op_read + lts__t_sectors_srcunit_tex_op_write) * 2 
    kernel_output.append(flits)
    # Average SM pipeline duty cycle, 34
    duty_cycle = sm__pipe_alu_cycles_active + sm__pipe_fma_cycles_active + sm__pipe_fp64_cycles_active + sm__pipe_shared_cycles_active + sm__pipe_tensor_cycles_active
    duty_cycle = duty_cycle / sm__cycles_elapsed_total / 5
    kernel_output.append(duty_cycle)
    # Average Num Idle SMs, 35
    kernel_output.append((1 - smsp__cycles_active/100) * 80)
    # Average active threads per warp, 36
    kernel_output.append(smsp__thread_inst_executed_per_inst_executed)
    # idc__requests_lookup_hit.sum, 37
    kernel_output.append(idc__requests_hit)
    # total cycle, 38
    kernel_output.append(sm__cycles_elapsed_total/80)
    # 39
    kernel_output.append(sm__sass_thread_inst_executed)

    # create numpy array
    kernel_output = np.array(kernel_output)
    kernel_output = np.transpose(kernel_output)

    unique_names = []
    accelsim_names = []
    output = []
    # get unique kernel names
    for kernel in kernel_names:
        kernel = kernel.replace(",","")
        kernel_plaintext = re.sub(r'\W+', '', kernel)
        if kernel not in unique_names and kernel_plaintext in kernel_name_map:
            unique_names.append(kernel)
            # accelsim_names.append(kernel_name_map.get(kernel_plaintext,""))
            accelsim_names.append(kernel_name_map.get(kernel_plaintext,kernel))
    
    apps = []
    # create array for each unique kernel
    # same kernel are combined
    for name in unique_names:
        apps.append(app_names[0])
    apps = np.array(apps)
    apps = np.reshape(apps, [-1,1])
    output = np.array(accelsim_names)
    output = np.reshape(output, [-1,1])
    output = np.append(apps,output,axis = 1)
    tmp_output = np.zeros_like(kernel_output[0])
    tmp_output = np.reshape(tmp_output,[1,-1])
    tmp_output = tmp_output[1:,1:]
    # print("total kernel combined in last app",kernel_names.shape[0])
    for name in unique_names:
        # replace all symbols and spaces
        name_text = re.sub(r'\W+', '', name)
        combine_kernel = np.zeros_like(kernel_output[0])
        combine_kernel = np.reshape(combine_kernel,[1,-1])
        count = 0
        for kernel in range(0,kernel_names.shape[0]):
            # combine_kernel = np.append(combine_kernel,np.reshape(kernel_output[kernel],[1,-1]),axis = 0)
            # replace all symbols and spaces to compare alphabets only
            kernel_text = re.sub(r'\W+', '', kernel_names[kernel])
            if name_text in kernel_text:
            # if name.split("(")[0] in kernel_names[kernel] or \
            #     name.split("<")[0] in kernel_names[kernel]:
                combine_kernel = np.append(combine_kernel,np.reshape(kernel_output[kernel],[1,-1]),axis = 0)
                count = count + 1
        combine_kernel = combine_kernel[1:]
        # print("kernel", "<" + kernel_names[kernel] + ">","found",count,"times")

        if(combine_kernel.shape[0]) is 0:
            # parsing error or string comparision error
            exit("ERROR,", kernel_names[kernel], "not found")

        print("#num_kernels: ",count)
        # calcualte elapsed cycle (total elapsed cycle / # of kernel launch)
        combine_kernel[:,38] = combine_kernel[:,38].astype(float) / float(count)
        # Num_Idle_SMs = Num_Idle_SMs / # of kernel launch
        # combine_kernel[:,35] = combine_kernel[:,35].astype(float) / float(count)
        # tmp = np.reshape(np.zeros_like(combine_kernel[0]),[1,-1])
        # calculate metrics that need to be added for the same kernel
        tmp_sum = combine_kernel[:,1:34].astype(float) / float(count)
        tmp_sum = tmp_sum.sum(axis=0)
        tmp_sum = np.reshape(tmp_sum, [1,-1])
        # calculate metrics that need to be averages for the same kernel
        tmp_avg = combine_kernel[:,34:37].astype(float)
        tmp_avg = tmp_avg.mean(axis=0)
        tmp_avg = np.reshape(tmp_avg, [1,-1])
        tmp = np.append(tmp_sum,tmp_avg,axis = 1)
        # sum again
        tmp_sum = combine_kernel[:,37:].astype(float)
        tmp_sum = tmp_sum.sum(axis=0)
        tmp_sum = np.reshape(tmp_sum, [1,-1])
        tmp = np.append(tmp,tmp_sum,axis = 1)
        tmp_output = np.append(tmp_output,tmp, axis = 0)
    # add kernel to the total array (for all apps)
    output = np.append(output,tmp_output,axis = 1)
    total = np.append(total,output,axis = 0)

# first line are all 0s when creating the array using np.zeros
# so remove first line
total = total[1:]
# un-comment these to save all metrics (used for verification)
# np.savetxt("output.csv",total,delimiter=',',fmt='%s')
# exit()


# select metrics used in Accel-Wattch
# these constants needs to be changed if new metrics are added
selected = np.empty((total.shape[0],0))
selected = np.append(selected,total[:,0:2],axis = 1)
selected = np.append(selected,total[:,6:12],axis = 1)
selected = np.append(selected,total[:,27:29],axis = 1)
selected = np.append(selected,total[:,30:37],axis = 1)
selected = np.append(selected,total[:,39:40],axis = 1)
# selected = np.append(selected,total[:,30:31],axis = 1)
# print(selected[0])

# add chip voltage - all 1 volts
ones = np.ones((selected.shape[0],1))
selected = np.concatenate((selected,ones),1)

# add csv titles
csv_titles = ["Benchmark", "Kernel", "L1_RH","L1_RM","L1_WH","L1_WM","CC_ACC","SHRD_ACC","DRAM_Rd","DRAM_Wr","L2_RH","L2_RM","L2_WH","L2_WM","NOC","Pipeline_Duty","Num_Idle_SMs","Elapsed_Cycles","Chip Voltage"]
csv_titles = np.array(csv_titles)
csv_titles = np.reshape(csv_titles,(1,19))
selected = np.concatenate((csv_titles,selected),0)


# save output
np.savetxt("hw_perf.csv",selected,delimiter=',',fmt='%s')