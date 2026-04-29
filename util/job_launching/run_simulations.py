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
from subprocess import Popen, PIPE

import sys
import re
import shutil
import glob
import datetime
import yaml
import common

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"


def parse_kernelslist(kernelslist_path):
    """Parse kernelslist.g and return list of kernel trace filenames.

    Filters out MemcpyHtoD and other non-kernel commands, returning only
    lines that start with 'kernel-'.
    """
    kernels = []
    with open(kernelslist_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("kernel-"):
                kernels.append(line)
    return kernels


# This function will pull the SO name out of the shared object,
# which will have current GIT commit number attatched.
def extract_version(exec_path, simulator):
    if simulator == "gpgpusim":
        regex_base = "gpgpu-sim_git-commit"
    else:
        regex_base = "accelsim-commit"
    regex_str = r".*({0}[^\s^']+).*".format(regex_base)
    strings_process = Popen(["strings", exec_path], stdout=PIPE)
    grep_process = Popen(
        ["grep", regex_base], stdin=strings_process.stdout, stdout=PIPE
    )
    strings_process.stdout.close()
    out, err = grep_process.communicate()
    version = re.sub(regex_str, r"\1", str(out.rstrip()))
    return version


#######################################################################################
# Class the represents each configuration you are going to run
# For example, if your sweep file has 2 entries 32k-L1 and 64k-L1 there will be 2
# ConfigurationSpec classes and the run_subdir name for each will be 32k-L1 and 64k-L1
# respectively
class ConfigurationSpec:
    #########################################################################################
    # Public Interface methods
    #########################################################################################
    # Class is constructed with a single line of text from the sweep_param file
    def __init__(self, nameTuple):
        name, params, config_file = nameTuple
        self.run_subdir = name
        self.params = params
        self.config_file = config_file

    def my_print(self):
        print("Run Subdir = " + self.run_subdir)
        print("Parameters = " + self.params)
        print("Base config file = " + self.config_file)

    def run(self, build_handle, benchmarks, run_directory, cuda_version, simdir):
        for dir_bench in benchmarks:
            exec_dir, data_dir, benchmark, self.command_line_args_list = dir_bench
            full_exec_dir = ""  # For traces it is not necessary to have the apps built
            full_data_dir = ""
            if options.trace_dir == "":
                full_exec_dir = common.dir_option_test(
                    os.path.expandvars(exec_dir), "", this_directory
                )
                try:
                    full_data_dir = common.dir_option_test(
                        os.path.join(
                            os.path.expandvars(data_dir), benchmark.replace("/", "_")
                        ),
                        "",
                        this_directory,
                    )
                except common.PathMissing:
                    pass

            self.benchmark_args_subdirs = {}
            for argmap in self.command_line_args_list:
                args = argmap["args"]
                self.benchmark_args_subdirs[args] = common.get_argfoldername(argmap)

            for argmap in self.command_line_args_list:
                args = argmap["args"]
                mem_usage = argmap["accel-sim-mem"]
                appargs_run_subdir = os.path.join(
                    benchmark.replace("/", "_"), self.benchmark_args_subdirs[args]
                )
                this_run_dir = os.path.join(
                    run_directory, appargs_run_subdir, self.run_subdir
                )

                # Determine profiling modes to run (always include a normal
                # unprofiled run alongside any profiled runs)
                profile_modes = [None]
                if options.profile_perf:
                    profile_modes.append("perf")
                if options.profile_mem:
                    profile_modes.append("mem")

                # Handle per-kernel mode
                if options.per_kernel and options.trace_dir != "":
                    benchmark_trace_dir = self.find_benchmark_trace_dir(appargs_run_subdir)
                    kernelslist_path = os.path.join(benchmark_trace_dir, "kernelslist.g")
                    kernels = parse_kernelslist(kernelslist_path)

                    for kernel_filename in kernels:
                        # Extract kernel name (e.g., "kernel-1" from "kernel-1-ctx_xxx.traceg.xz")
                        kernel_name = kernel_filename.split("-ctx_")[0]
                        if "-ctx_" not in kernel_filename:
                            # Fallback: use everything before .traceg
                            kernel_name = kernel_filename.split(".traceg")[0]

                        kernel_run_dir = os.path.join(this_run_dir, "per-kernel", kernel_name)
                        self.setup_per_kernel_directory(kernel_run_dir, kernel_filename, benchmark_trace_dir)
                        self.append_gpgpusim_config(
                            benchmark, kernel_run_dir, appargs_run_subdir, self.config_file
                        )

                        for profile_mode in profile_modes:
                            sim_tmpl = self.text_replace_torque_sim(
                                full_data_dir,
                                kernel_run_dir,
                                benchmark,
                                cuda_version,
                                args,
                                simdir,
                                full_exec_dir,
                                build_handle,
                                mem_usage,
                                kernel_name=kernel_name,
                                profile_mode=profile_mode,
                            )
                            # Submit the job
                            self._submit_job(
                                kernel_run_dir, benchmark, args, build_handle,
                                kernel_name=kernel_name, sim_template=sim_tmpl
                            )
                else:
                    # Standard mode: run all kernels together
                    self.setup_run_directory(
                        full_data_dir, this_run_dir, data_dir, appargs_run_subdir
                    )
                    self.append_gpgpusim_config(
                        benchmark, this_run_dir, appargs_run_subdir, self.config_file
                    )

                    for profile_mode in profile_modes:
                        sim_tmpl = self.text_replace_torque_sim(
                            full_data_dir,
                            this_run_dir,
                            benchmark,
                            cuda_version,
                            args,
                            simdir,
                            full_exec_dir,
                            build_handle,
                            mem_usage,
                            profile_mode=profile_mode,
                        )
                        # Submit the job
                        self._submit_job(this_run_dir, benchmark, args, build_handle,
                                         sim_template=sim_tmpl)
            self.benchmark_args_subdirs.clear()

    def _submit_job(self, run_dir, benchmark, args, build_handle, kernel_name=None, sim_template=None):
        """Submit a job to the scheduler and log it."""
        if options.no_launch:
            return

        torque_out_filename = this_directory + "torque_out.{0}.txt".format(os.getpid())
        torque_out_file = open(torque_out_filename, "w+")
        saved_dir = os.getcwd()
        os.chdir(run_dir)
        if (
            subprocess.call(
                [job_submit_call, os.path.join(run_dir, sim_template or job_template)],
                stdout=torque_out_file,
            )
            < 0
        ):
            exit("Error Launching Job")
        else:
            # Parse the torque output for just the numeric ID
            torque_out_file.seek(0)
            torque_out = re.sub(
                r"[^\d]*(\d*).*", r"\1", torque_out_file.read().strip()
            )
            job_desc = benchmark + "-" + self.benchmark_args_subdirs[args] + " " + self.run_subdir
            if kernel_name:
                job_desc += "/" + kernel_name
            print("Job " + torque_out + " queued (" + job_desc + ")")
        torque_out_file.close()
        os.remove(torque_out_filename)
        os.chdir(saved_dir)

        if len(torque_out) > 0:
            # Dump the benchmark description to the logfile
            if not os.path.exists(this_directory + "logfiles/"):
                try:
                    os.makedirs(this_directory + "logfiles/")
                except:
                    pass
            now_time = datetime.datetime.now()
            day_string = now_time.strftime("%y.%m.%d-%A")
            time_string = now_time.strftime("%H:%M:%S")
            log_name = "sim_log.{0}".format(options.launch_name)
            logfile = open(
                this_directory + "logfiles/" + log_name + "." + day_string + ".txt",
                "a",
            )
            log_entry = self.benchmark_args_subdirs[args]
            if kernel_name:
                log_entry += "/" + kernel_name
            print(
                "%s %6s %-22s %-100s %-25s %s"
                % (
                    time_string,
                    torque_out,
                    benchmark,
                    log_entry,
                    self.run_subdir,
                    build_handle,
                ),
                file=logfile,
            )
            logfile.close()

    #########################################################################################
    # Internal utility methods
    #########################################################################################

    def find_benchmark_trace_dir(self, appargs_subdir):
        """Find the benchmark trace directory for the given appargs_subdir."""
        benchmark_trace_dir = None
        paths_to_try = (
            [os.path.join(options.trace_dir, appargs_subdir, "traces")]
            + glob.glob(
                os.path.join(
                    options.trace_dir, "**", "**", appargs_subdir, "traces"
                )
            )
            + glob.glob(
                os.path.join(options.trace_dir, "**", appargs_subdir, "traces")
            )
        )
        for path in paths_to_try:
            try:
                benchmark_trace_dir = common.dir_option_test(
                    path, "", this_directory
                )
                break
            except common.PathMissing as e:
                pass

        if benchmark_trace_dir == None:
            sys.exit(
                "Cannot find traces in any of the paths: {0}".format(paths_to_try)
            )
        return os.path.abspath(benchmark_trace_dir)

    # copies and links the necessary files to the run directory
    def setup_run_directory(
        self, full_data_dir, this_run_dir, data_dir, appargs_subdir
    ):
        if not os.path.isdir(this_run_dir):
            os.makedirs(this_run_dir)

        files_to_copy_to_run_dir = (
            glob.glob(os.path.join(full_data_dir, "*.ptx"))
            + glob.glob(os.path.join(full_data_dir, "*.cl"))
            + glob.glob(os.path.join(full_data_dir, "*.h"))
            + glob.glob(os.path.dirname(self.config_file) + "/*.icnt")
            + glob.glob(os.path.dirname(self.config_file) + "/*.csv")
            + glob.glob(os.path.dirname(self.config_file) + "/*.xml")
        )

        for file_to_cp in files_to_copy_to_run_dir:
            new_file = os.path.join(
                this_run_dir, os.path.basename(this_directory + file_to_cp)
            )
            if os.path.isfile(new_file):
                os.remove(new_file)
            shutil.copyfile(file_to_cp, new_file)

        # link the data directory
        benchmark_data_dir = os.path.join(full_data_dir, "data")
        if os.path.isdir(benchmark_data_dir):
            if os.path.lexists(os.path.join(this_run_dir, "data")):
                os.remove(os.path.join(this_run_dir, "data"))
            os.symlink(benchmark_data_dir, os.path.join(this_run_dir, "data"))

        # link the traces directory
        if options.trace_dir != "":
            benchmark_trace_dir = self.find_benchmark_trace_dir(appargs_subdir)
            if os.path.isdir(benchmark_trace_dir):
                if os.path.lexists(os.path.join(this_run_dir, "traces")):
                    os.remove(os.path.join(this_run_dir, "traces"))
                os.symlink(benchmark_trace_dir, os.path.join(this_run_dir, "traces"))

        all_data_link = os.path.join(this_run_dir, "data_dirs")
        if os.path.lexists(all_data_link):
            os.remove(all_data_link)
        if os.path.exists(os.path.join(this_directory, data_dir)):
            os.symlink(os.path.join(this_directory, data_dir), all_data_link)

    def setup_per_kernel_directory(self, kernel_run_dir, kernel_filename, benchmark_trace_dir):
        """Set up a per-kernel run directory with its own traces folder and kernelslist.g."""
        # Create directory structure
        if not os.path.isdir(kernel_run_dir):
            os.makedirs(kernel_run_dir)

        traces_dir = os.path.join(kernel_run_dir, "traces")
        if not os.path.isdir(traces_dir):
            os.makedirs(traces_dir)

        # Write kernelslist.g with just this kernel
        kernelslist_path = os.path.join(traces_dir, "kernelslist.g")
        with open(kernelslist_path, "w") as f:
            f.write(kernel_filename + "\n")

        # Symlink the kernel trace file
        src_trace = os.path.join(benchmark_trace_dir, kernel_filename)
        dst_trace = os.path.join(traces_dir, kernel_filename)
        if os.path.lexists(dst_trace):
            os.remove(dst_trace)
        if os.path.exists(src_trace):
            os.symlink(src_trace, dst_trace)

        # Copy config files (.icnt, .csv, .xml)
        files_to_copy = (
            glob.glob(os.path.dirname(self.config_file) + "/*.icnt")
            + glob.glob(os.path.dirname(self.config_file) + "/*.csv")
            + glob.glob(os.path.dirname(self.config_file) + "/*.xml")
        )
        for file_to_cp in files_to_copy:
            new_file = os.path.join(kernel_run_dir, os.path.basename(file_to_cp))
            if os.path.isfile(new_file):
                os.remove(new_file)
            shutil.copyfile(file_to_cp, new_file)

    # replaces all the "REAPLCE_*" strings in the .sim file
    def text_replace_torque_sim(
        self,
        full_run_dir,
        this_run_dir,
        benchmark,
        cuda_version,
        command_line_args,
        libpath,
        exec_dir,
        gpgpusim_build_handle,
        mem_usage,
        kernel_name=None,
        profile_mode=None,
    ):
        # get the pre-launch sh commands
        prelaunch_filename = full_run_dir + "benchmark_pre_launch_command_line.txt"
        benchmark_command_line = ""
        if os.path.isfile(prelaunch_filename):
            f = open(prelaunch_filename)
            benchmark_command_line = f.read().strip()
            f.close()

        if options.trace_dir == "":
            # If the config contains "SASS" and you have not specified the trace directory, then likely something is wrong
            if ("SASS" in self.run_subdir):
                print("You are trying to run a configuration with SASS in it, but have not specified a trace directory."+\
                      " If you want to run SASS traces, please specify -T to point to the top-level trace directory")
                exit(1)
            exec_name = (
                options.benchmark_exec_prefix
                + " "
                + os.path.join(this_directory, exec_dir, benchmark)
            )
        else:
            exec_name = (
                options.benchmark_exec_prefix
                + " "
                + os.path.join(libpath, "accel-sim.out")
            )

        # Wrap with profiling tool based on profile_mode
        if profile_mode == "perf":
            perf_out = os.path.join(this_run_dir, common.PROFILE_PERF_DATA_FILE)
            exec_name = "perf record -g -o {0} -- {1}".format(
                perf_out, exec_name
            )
        elif profile_mode == "mem":
            ht_out = os.path.join(this_run_dir, common.PROFILE_HEAPTRACK_FILE)
            exec_name = "{0} --record-only -o {1} {2}".format(
                options.heaptrack_bin, ht_out, exec_name
            )

        # Test the existance of required env variables
        if str(os.getenv("GPGPUSIM_ROOT")) == "None":
            exit("\nERROR - Specify GPGPUSIM_ROOT prior to running this script")
        if str(os.getenv("OPENCL_REMOTE_GPU_HOST")) == "None":
            os.environ["OPENCL_REMOTE_GPU_HOST"] = ""
        if str(os.getenv("GPGPUSIM_CONFIG")) == "None":
            exit("\nERROR - Specify GPGPUSIM_CONFIG prior to running this script")

        # If the user specified the memory use that
        if options.job_mem != None:
            mem_usage = options.job_mem
        # if we are using PTX (GPGPU-Sim) - then just assume 4G
        elif options.trace_dir == "":
            mem_usage = "4G"
        if options.trace_dir == "":
            if command_line_args == None:
                txt_args = ""
            else:
                txt_args = str(command_line_args)
        else:
            txt_args = " -config ./gpgpusim.config -trace ./traces/kernelslist.g"

        if os.getenv("TORQUE_QUEUE_NAME") == None:
            queue_name = "batch"
        else:
            queue_name = os.getenv("TORQUE_QUEUE_NAME")

        # do the text replacement for the .sim file
        sim_name = benchmark + "-" + self.benchmark_args_subdirs[command_line_args] + "." +\
                                gpgpusim_build_handle
        if kernel_name:
            sim_name += "." + kernel_name
        # Add postfix for profiling runs so output files don't overwrite each other
        if profile_mode:
            sim_name += ".{0}".format(profile_mode)
        # Truncate long simulation file names
        sim_name = sim_name[:200]
        replacement_dict = {"NAME":sim_name,
                            "NODES":"1", 
                            "GPGPUSIM_ROOT":os.getenv("GPGPUSIM_ROOT"),
                            "LIBPATH": libpath,
                            "SUBDIR":this_run_dir,
                            "OPENCL_REMOTE_GPU_HOST":os.getenv("OPENCL_REMOTE_GPU_HOST"),
                            "BENCHMARK_SPECIFIC_COMMAND":benchmark_command_line,
                            "PATH":os.getenv("PATH"),
                            "EXEC_NAME":exec_name,
                            "QUEUE_NAME":queue_name,
                            "COMMAND_LINE":txt_args,
                            "MEM_USAGE": mem_usage
                            }
        torque_text = open(this_directory + job_template).read().strip()
        for entry in replacement_dict:
            torque_text = re.sub(
                "REPLACE_" + entry, str(replacement_dict[entry]), torque_text
            )
        # Use mode-specific filenames when profiling to avoid overwrites
        suffix = ".{0}".format(profile_mode) if profile_mode else ""
        sim_template = job_template.replace(".sim", suffix + ".sim") if suffix else job_template
        open(os.path.join(this_run_dir, sim_template), "w").write(torque_text)
        justrunfile = os.path.join(this_run_dir, "justrun{0}.sh".format(suffix))
        open(justrunfile, 'w').write(exec_name + " " + txt_args + " | tee gpgpu-sim-out_`date '+%b_%d_%H:%M.%S'`.txt")
        os.chmod(justrunfile, 0o744)
        return sim_template

    # replaces all the "REPLACE_*" strings in the gpgpusim.config file
    def append_gpgpusim_config(
        self, bench_name, this_run_dir, appargs_run_subdir, config_text_file
    ):
        benchmark_spec_opts_file = os.path.expandvars(
            os.path.join(
                "$GPUAPPS_ROOT",
                "benchmarks",
                "app-specific-gpgpu-sim-options",
                bench_name,
                "benchmark_options.txt",
            )
        )
        benchmark_spec_opts = ""
        if os.path.isfile(benchmark_spec_opts_file):
            f = open(benchmark_spec_opts_file)
            benchmark_spec_opts = f.read().strip()
            f.close()

        config_text = open(config_text_file).read()
        config_text += "\n" + benchmark_spec_opts + "\n" + self.params + "\n"

        if options.accelwattch_HW:
            # if bench_name == "cutlass_perf_test":
            #     config_text += "\n" + "-hw_perf_bench_name " + appargs_run_subdir.replace('/','_') + "\n"
            # else:
            config_text += "\n" + "-hw_perf_bench_name " + bench_name + "\n"

        if options.trace_dir != "":
            cfgsubdir = re.sub(r".*(configs.*)gpgpusim.config", r"\1", config_text_file)
            config_text += "\n" + "# Accel-Sim Parameters" + "\n"
            accelsim_cfg = os.path.expandvars(
                os.path.join("$ACCELSIM_ROOT", cfgsubdir, "trace.config")
            )
            config_text += open(accelsim_cfg).read()

        open(os.path.join(this_run_dir, "gpgpusim.config"), "w").write(config_text)


# -----------------------------------------------------------
# main script start
# -----------------------------------------------------------
(options, args) = common.parse_run_simulations_options()

if str(os.getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN")) != "1":
    sys.exit("ERROR - Please run setup_environment before running this script")


cuda_version = common.get_cuda_version(this_directory)

if options.run_directory == "":
    options.run_directory = os.path.join(
        this_directory, "../../sim_run_%s" % cuda_version
    )
else:
    options.run_directory = os.path.join(os.getcwd(), options.run_directory)

# Let's copy out the .so file so that builds don't interfere with running tests
# If the user does not specify a so file, then use the one in the git repo and copy it out.
if options.trace_dir == "":
    options.simulator_dir = common.dir_option_test(
        options.simulator_dir,
        os.path.join(os.getenv("GPGPUSIM_ROOT"), "lib", os.getenv("GPGPUSIM_CONFIG")),
        this_directory,
    )
    simulator_path = os.path.join(options.simulator_dir, "libcudart.so")
else:
    options.simulator_dir = common.dir_option_test(
        options.simulator_dir,
        os.path.join(os.getenv("ACCELSIM_ROOT"), "bin", os.getenv("ACCELSIM_CONFIG")),
        this_directory,
    )
    simulator_path = os.path.join(options.simulator_dir, "accel-sim.out")

if options.trace_dir == "":
    version_string = extract_version(simulator_path, "gpgpusim")
else:
    gpgpusim_path = os.path.join(
        os.getenv("GPGPUSIM_ROOT"), "lib", os.getenv("GPGPUSIM_CONFIG"), "libcudart.so"
    )
    version_string = extract_version(simulator_path, "accelsim") + extract_version(
        gpgpusim_path, "gpgpusim"
    )
running_sim_dir = os.path.join(
    options.run_directory, "gpgpu-sim-builds", version_string
)
if not os.path.exists(running_sim_dir):
    # In the very rare case that concurrent builds try to make the directory at the same time
    # (after the test to os.path.exists -- this has actually happened...)
    try:
        os.makedirs(running_sim_dir)
    except:
        pass

cached_sim_path = os.path.join(running_sim_dir, os.path.basename(simulator_path))
if not os.path.exists(cached_sim_path) or os.path.getmtime(simulator_path) > os.path.getmtime(cached_sim_path):
    shutil.copy(simulator_path, running_sim_dir)
options.simulator_dir = running_sim_dir

common.load_defined_yamls()

# Test for the existance of a cluster management system
job_submit_call = None
job_template = None
if options.launcher != "":
    if options.launcher == "qsub":
        job_submit_call = options.launcher
        job_template = "torque.sim"
    elif options.launcher == "sbatch":
        job_submit_call = options.launcher
        job_template = "slurm.sim"
    elif options.launcher == "local":
        job_submit_call = os.path.join(this_directory, "procman.py")
        job_template = "slurm.sim"
elif any(
    [
        os.path.isfile(os.path.join(p, "sbatch"))
        for p in os.getenv("PATH").split(os.pathsep)
    ]
):
    job_submit_call = "sbatch"
    job_template = "slurm.sim"
elif any(
    [
        os.path.isfile(os.path.join(p, "qsub"))
        for p in os.getenv("PATH").split(os.pathsep)
    ]
):
    job_submit_call = "qsub"
    job_template = "torque.sim"
else:
    print("Cannot find a supported job management system. Spawning jobs locally.")
    job_submit_call = os.path.join(this_directory, "procman.py")
    job_template = "slurm.sim"

if not any(
    [
        os.path.isfile(os.path.join(p, "nvcc"))
        for p in os.getenv("PATH").split(os.pathsep)
    ]
):
    exit(
        "ERROR - Cannot find nvcc PATH... Is CUDA_INSTALL_PATH/bin in the system PATH?"
    )

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cfgs = common.gen_configs_from_list(options.configs_list.split(","))
configurations = []
for config in cfgs:
    configurations.append(ConfigurationSpec(config))

print(
    "Running Simulations with GPGPU-Sim built from \n{0}\n ".format(version_string)
    + "\nUsing configs: "
    + options.configs_list
    + "\nBenchmark: "
    + options.benchmark_list
)

for config in configurations:
    config.my_print()
    config.run(
        version_string,
        benchmarks,
        options.run_directory,
        cuda_version,
        options.simulator_dir,
    )

if "procman" in job_submit_call and not options.no_launch:
    if options.cores == None:
        subprocess.call([job_submit_call, "-S"])
    else:
        subprocess.call([job_submit_call, "-S", "-c", options.cores])
