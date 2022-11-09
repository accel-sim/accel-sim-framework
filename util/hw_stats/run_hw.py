#!/usr/bin/env python3

# Copyright (c) 2018-2021, Mahmoud Khairy, Vijay Kandiah, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
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


from optparse import OptionParser
import os
import subprocess
import os
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
import sys
sys.path.insert(0,os.path.join(this_directory,"..","job_launching"))
import common
import re
import shutil
import glob
import datetime
import yaml
import common
import re
import datetime

# We will look for the benchmarks 
parser = OptionParser()
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                 help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for " +\
                       "the benchmark suite names.",
                 default="rodinia_2.0-ft")
parser.add_option("-D", "--device_num", dest="device_num",
                 help="CUDA device number",
                 default="0")
parser.add_option("-n", "--norun", dest="norun", action="store_true",
                 help="Do not actually run the apps, just create the dir structure and launch files")
parser.add_option("-R", "--repeat_cycle", dest="repeat_cycle", default=1,
                 help="When running the cycle tests, do them this many times (good when DVFS is enabled)")
parser.add_option("-N", "--nsight_profiler", dest="nsight_profiler", action="store_true",
                 help="use the new nsight cli profiler")
parser.add_option("-d", "--disable_nvprof", dest="disable_nvprof", action="store_true",
                 help="do not use nvprof (decrecated in Turing+)")
parser.add_option("-S", "--nsys_profiler", dest="nsys_profiler", action="store_true",
                 help="Use the Nsys profiler for counting cycles instead of Ncu")
parser.add_option("-l", "--limit_kernel_number", dest="kernel_number", type=int, default=-99,
                 help="Limits the number of profiled kernels (useful in larger applications")
parser.add_option("-C", "--collect", dest="collect", default="cycles",
                help="Pass what you want from the hardware. Options are: \"cycles,other_stats\"")

(options, args) = parser.parse_args()

if not options.disable_nvprof:
    if not any([os.path.isfile(os.path.join(p, "nvprof")) for p in os.getenv("PATH").split(os.pathsep)]):
        exit("ERROR - Cannot find nv-nsight-cu-cli PATH... Is CUDA_INSTALL_PATH/bin in the system PATH?")

if options.nsight_profiler:
    if not any([os.path.isfile(os.path.join(p, "nv-nsight-cu-cli")) for p in os.getenv("PATH").split(os.pathsep)]):
        exit("ERROR - Cannot find nv-nsight-cu-cli PATH... Is CUDA_INSTALL_PATH/bin in the system PATH?")

common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cuda_version = common.get_cuda_version( this_directory )
now_time = datetime.datetime.now()
day_string = now_time.strftime("%y.%m.%d-%A")
time_string = now_time.strftime("%H:%M:%S")
logfile = day_string + "--" + time_string + ".csv"

for bench in benchmarks:
    edir, ddir, exe, argslist = bench
    for argpair in argslist:
        args = argpair["args"]
        run_name = os.path.join( exe, common.get_argfoldername( args ) )

        this_run_dir = os.path.join(this_directory, "..", "..", "hw_run", "device-" + options.device_num, cuda_version, run_name)
        if not os.path.exists(this_run_dir):
            os.makedirs(this_run_dir)

        # link the data directory
        try:
            benchmark_data_dir = common.dir_option_test(os.path.join(ddir,exe,"data"),"",this_directory)
            if os.path.lexists(os.path.join(this_run_dir, "data")):
                os.remove(os.path.join(this_run_dir, "data"))
            os.symlink(benchmark_data_dir, os.path.join(this_run_dir,"data"))
        except common.PathMissing:
            pass

        all_data_link = os.path.join(this_run_dir,"data_dirs")
        if os.path.lexists(all_data_link):
            os.remove(all_data_link)
        top_data_dir_path = common.dir_option_test(ddir, "", this_directory)
        os.symlink(top_data_dir_path, all_data_link)

        if args == None:
            args = ""

        exec_path = common.file_option_test(os.path.join(edir, exe),"",this_directory)
        sh_contents = ""
        kernel_number = ""
        if('mlperf' in exec_path):
            exec_path = "sh " + exec_path
            # For MLPerf we are by default limiting the number of profiled kernels to 1000
            # This can be overriden by explicitly indicating the number of kernels via the -c argument.
            kernel_number = ' -c 1000 '
        if(options.kernel_number > 0):
            kernel_number = ' -c '+str(options.kernel_number)+' '

        if "other_stats" in options.collect:
            if not options.disable_nvprof:
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 30m nvprof --concurrent-kernels off --print-gpu-trace -u us --metrics all --demangling off --csv --log-file " +\
                    os.path.join(this_run_dir,logfile) + " " + exec_path + " " + str(args) + " "
            if options.nsight_profiler:
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 30m nv-nsight-cu-cli --metrics gpc__cycles_elapsed.avg,sm__cycles_elapsed.sum,smsp__inst_executed.sum," +\
                    "sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum," +\
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_srcunit_tex_op_read.sum,"+\
                    "lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum," +\
                    "lts__t_sector_op_write_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum.per_second,dram__sectors_read.sum,dram__sectors_write.sum,dram__bytes_read.sum," +\
                    "sm__inst_executed.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum," +\
                    ",l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum,idc__requests.sum,idc__requests_lookup_hit.sum," +\
                    "sm__sass_inst_executed_op_shared_ld.sum,sm__sass_inst_executed_op_shared_st.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum,sm__pipe_alu_cycles_active.sum,sm__pipe_fma_cycles_active.sum,sm__pipe_fp64_cycles_active.sum,sm__pipe_shared_cycles_active.sum,sm__pipe_tensor_cycles_active.sum,sm__pipe_tensor_op_hmma_cycles_active.sum,sm__cycles_active.sum,sm__cycles_active.avg,sm__cycles_elapsed.avg,sm__sass_thread_inst_executed_op_integer_pred_on.sum,sm__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on.sum,sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum,sm__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fp16.sum,sm__inst_executed_pipe_fp64.sum,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_tex.sum,sm__inst_executed_pipe_xu.sum,sm__inst_executed_pipe_lsu.sum," +\
                    "sm__sass_thread_inst_executed_op_fp16_pred_on.sum,sm__sass_thread_inst_executed_op_fp32_pred_on.sum,sm__sass_thread_inst_executed_op_fp64_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_inst_executed_op_memory_128b.sum,sm__sass_inst_executed_op_memory_64b.sum,sm__sass_inst_executed_op_memory_32b.sum,sm__sass_inst_executed_op_memory_16b.sum,sm__sass_inst_executed_op_memory_8b.sum,smsp__thread_inst_executed_per_inst_executed.ratio,sm__sass_thread_inst_executed.sum" +\
                    " --csv --page raw --target-processes all " + kernel_number +\
                    " " + exec_path + " " + str(args) +\
                    " | tee " + os.path.join(this_run_dir,logfile + ".nsight")

        for i in range(int(options.repeat_cycle)):
            if not options.disable_nvprof:
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 5m nvprof --unified-memory-profiling off --concurrent-kernels off --print-gpu-trace -u us --demangling off --csv --log-file " +\
                    os.path.join(this_run_dir,logfile + ".cycle.{0}".format(i)) + " " + exec_path + " " + str(args)
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 5m nvprof --concurrent-kernels off --print-gpu-trace --events elapsed_cycles_sm --demangling off --csv --log-file " +\
                    os.path.join(this_run_dir,logfile + ".elapsed_cycles_sm.{0}".format(i)) + " " + exec_path + " " + str(args) + " "
            if options.nsys_profiler:
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 5m nsys profile -o "+os.path.join(this_run_dir,"out") + " " +\
                        exec_path + " " + str(args) + "; nsys stats -f csv --report gputrace "+os.path.join(this_run_dir,"out.qdrep")+\
                        " | tee cycles.csv; python "+os.path.join(this_directory,"postprocess-nsys-csv.py")+" --path "+this_run_dir
            elif options.nsight_profiler:
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 5m nv-nsight-cu-cli --target-processes all --metrics gpc__cycles_elapsed.avg --csv " +\
                        exec_path + " " + str(args) + " | tee " +\
                        os.path.join(this_run_dir,logfile + ".gpc__cycles_elapsed.{0}".format(i))

        open(os.path.join(this_run_dir,"run.sh"), "w").write(sh_contents)
        if subprocess.call(['chmod', 'u+x', os.path.join(this_run_dir,"run.sh")]) != 0:
            exit("Error chmod runfile")

        if not options.norun:
            saved_dir = os.getcwd()
            os.chdir(this_run_dir)
            print("Running {0}".format(exe))

            if subprocess.call(["bash", "run.sh"]) != 0:
                print("Error invoking profiler on {0}".format(this_run_dir))
            os.chdir(saved_dir)