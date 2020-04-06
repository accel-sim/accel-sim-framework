#!/usr/bin/env python

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
parser.add_option("-c", "--cycle_only", dest="cycle_only", action="store_true",
                 help="Just get Kernel Duration")
parser.add_option("-R", "--repeat_cycle", dest="repeat_cycle", default=1,
                 help="When running the cycle tests, do them this many times (good when DVFS is enabled)")
parser.add_option("-N", "--nsight_profiler", dest="nsight_profiler", action="store_true",
                 help="use the new nsight cli profiler")
parser.add_option("-d", "--disable_nvprof", dest="disable_nvprof", action="store_true",
                 help="do not use nvprof (decrecated in Turing+)")
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
    ddir = os.path.join(this_directory,ddir,exe)
    for args in argslist:
        run_name = os.path.join( exe, common.get_argfoldername( args ) )

        this_run_dir = os.path.join(this_directory, "..", "..", "run_hw", "device-" + options.device_num, cuda_version, run_name)
        if not os.path.exists(this_run_dir):
            os.makedirs(this_run_dir)

        # link the data directory
        if os.path.isdir(os.path.join(ddir, "data")):
            if os.path.lexists(os.path.join(this_run_dir, "data")):
                os.remove(os.path.join(this_run_dir, "data"))
            os.symlink(os.path.join(ddir, "data"), os.path.join(this_run_dir,"data"))

        if args == None:
            args = ""

        sh_contents = ""
        if not options.cycle_only:
            if not options.disable_nvprof:
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 5m nvprof --concurrent-kernels off --print-gpu-trace -u us --metrics all --demangling off --csv --log-file " +\
                    os.path.join(this_run_dir,logfile) + " " + os.path.join(this_directory, edir,exe) + " " + str(args) + " "
            if options.nsight_profiler:
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 5m nv-nsight-cu-cli --metrics gpc__cycles_elapsed.avg,sm__cycles_elapsed.sum,smsp__inst_executed.sum," +\
                    "sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum," +\
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_srcunit_tex_op_read.sum,"+\
                    "lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum," +\
                    "lts__t_sector_op_write_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum.per_second,dram__sectors_read.sum,dram__sectors_write.sum,dram__bytes_read.sum " +\
                    " --csv --page raw " +\
                    " " + os.path.join(this_directory, edir,exe) + " " + str(args) +\
                    " | tee " + os.path.join(this_run_dir,logfile + ".nsight")

        for i in range(int(options.repeat_cycle)):
            if not options.disable_nvprof:
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 5m nvprof --unified-memory-profiling off --concurrent-kernels off --print-gpu-trace -u us --demangling off --csv --log-file " +\
                    os.path.join(this_run_dir,logfile + ".cycle.{0}".format(i)) + " " + os.path.join(this_directory, edir,exe) + " " + str(args)
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 5m nvprof --concurrent-kernels off --print-gpu-trace --events elapsed_cycles_sm --demangling off --csv --log-file " +\
                    os.path.join(this_run_dir,logfile + ".elapsed_cycles_sm.{0}".format(i)) + " " + os.path.join(this_directory, edir,exe) + " " + str(args) + " "
            if options.nsight_profiler:
                sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num +\
                    "\" ; timeout 5m nv-nsight-cu-cli --metrics gpc__cycles_elapsed.avg --csv " +\
                        os.path.join(this_directory, edir,exe) + " " + str(args) + " | tee " +\
                        os.path.join(this_run_dir,logfile + ".gpc__cycles_elapsed.{0}".format(i))

        open(os.path.join(this_run_dir,"run.sh"), "w").write(sh_contents)
        if subprocess.call(['chmod', 'u+x', os.path.join(this_run_dir,"run.sh")]) != 0:
            exit("Error chmod runfile")

        if not options.norun:
            saved_dir = os.getcwd()
            os.chdir(this_run_dir)
            print "Running {0}".format(exe)

            if subprocess.call(["bash", "run.sh"]) != 0:
                print "Error invoking profiler on {0}".format(this_run_dir)
            os.chdir(saved_dir)
