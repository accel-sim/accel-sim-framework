#!/usr/bin/env python3

from optparse import OptionParser
import os
import subprocess
import sys
import datetime
import shutil

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.insert(0, os.path.join(this_directory, "..", "..", "job_launching"))
import common

parser = OptionParser()
parser.add_option(
    "-B",
    "--benchmark_list",
    dest="benchmark_list",
    help="a comma separated list of benchmark suites to run. See apps/define-*.yml for "
    + "the benchmark suite names.",
    default="rodinia_2.0-ft",
)
parser.add_option(
    "-D", "--device_num", dest="device_num", help="CUDA device number", default="0"
)
parser.add_option(
    "-n",
    "--norun",
    dest="norun",
    action="store_true",
    help="Do not actually run the apps, just create the dir structure and launch files",
)
parser.add_option(
    "-M",
    "--metrics",
    dest="metrics",
    default="sm__cycles_elapsed.avg,sm__inst_executed.sum",
    help="Comma-separated list of CUPTI metrics to collect",
)
parser.add_option(
    "-K",
    "--kernel_count",
    dest="kernel_count",
    type=int,
    default=1,
    help="Number of kernels per output CSV file",
)
parser.add_option(
    "-I",
    "--interval",
    dest="interval",
    type=int,
    default=3000,
    help="PM sampling interval in sysclk ticks",
)
parser.add_option(
    "-S",
    "--max_samples",
    dest="max_samples",
    type=int,
    default=80000,
    help="Maximum number of samples per session",
)
parser.add_option(
    "--hw_buffer",
    dest="hw_buffer",
    type=int,
    default=9388608000,
    help="Hardware buffer size in bytes",
)

(options, args) = parser.parse_args()

common.load_defined_yamls()

benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cuda_version = common.get_cuda_version(this_directory)
now_time = datetime.datetime.now()
day_string = now_time.strftime("%y.%m.%d-%A")
time_string = now_time.strftime("%H:%M:%S")

# Path to the CUPTI injection library
injection_lib = os.path.join(this_directory, "build", "libpmsampling_injection.so")
if not os.path.exists(injection_lib):
    sys.exit(f"Error: Injection library not found at {injection_lib}\nPlease build it first.")

for bench in benchmarks:
    edir, ddir, exe, argslist = bench
    for argpair in argslist:
        args = argpair["args"]
        run_name = os.path.join(exe, common.get_argfoldername(argpair))

        this_run_dir = os.path.join(
            this_directory,
            "..",
            "..",
            "..",
            "hw_run",
            "cupti",
            "device-" + options.device_num,
            cuda_version,
            run_name,
        )
        if not os.path.exists(this_run_dir):
            os.makedirs(this_run_dir)

        # link the data directory
        try:
            benchmark_data_dir = common.dir_option_test(
                os.path.join(ddir, exe, "data"), "", this_directory
            )
            if os.path.lexists(os.path.join(this_run_dir, "data")):
                os.remove(os.path.join(this_run_dir, "data"))
            os.symlink(benchmark_data_dir, os.path.join(this_run_dir, "data"))
        except common.PathMissing:
            pass

        all_data_link = os.path.join(this_run_dir, "data_dirs")
        if os.path.lexists(all_data_link):
            os.remove(all_data_link)
        top_data_dir_path = common.dir_option_test(ddir, "", this_directory)
        os.symlink(top_data_dir_path, all_data_link)

        if args is None:
            args = ""

        exec_path = common.file_option_test(os.path.join(edir, exe), "", this_directory)

        # Build the run script with CUPTI injection
        sh_contents = f"""#!/bin/bash

export CUDA_VISIBLE_DEVICES="{options.device_num}"
export CUDA_VERSION="{cuda_version}"

# CUPTI PM Sampling configuration
export INJECTION_METRICS="{options.metrics}"
export INJECTION_KERNEL_COUNT={options.kernel_count}
export PM_SAMPLING_INTERVAL_SYSCLK={options.interval}
export PM_SAMPLING_MAX_SAMPLES={options.max_samples}
export PM_SAMPLING_HW_BUFFER_BYTES={options.hw_buffer}
export CUDA_INJECTION64_PATH="{injection_lib}"

# Run the application
{exec_path} {args}
"""

        run_script = os.path.join(this_run_dir, "run.sh")
        with open(run_script, "w") as f:
            f.write(sh_contents)
        subprocess.call(["chmod", "u+x", run_script])

        if not options.norun:
            saved_dir = os.getcwd()
            os.chdir(this_run_dir)
            print(f"Running {exe} with args: {args}")
            print(f"  Output directory: {this_run_dir}")

            ret = subprocess.call(["bash", "run.sh"])
            if ret != 0:
                print(f"Error running {exe}")

            os.chdir(saved_dir)
        else:
            print(f"Created run script: {run_script}")
