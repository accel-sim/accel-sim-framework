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
import subprocess
import re
import os
import yaml
import glob
import hashlib

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

defined_apps = {}
defined_baseconfigs = {}
defined_xtracfgs = {}


def get_argfoldername(args):
    if args == "" or args == None:
        return "NO_ARGS"
    else:
        foldername = re.sub(r"[^a-z^A-Z^0-9]", "_", str(args).strip())
        # For every long arg lists - create a hash of the input args
        if len(str(args)) > 256:
            foldername = "hashed_args_" + hashlib.md5(args).hexdigest()
        return foldername


# Test to see if the passed config adheres to any defined configs and add it to the configrations to run/collect.
def get_config(name, defined_baseconfigs, defined_xtracfgs):
    tokens = name.split("-")
    if tokens[0] not in defined_baseconfigs:
        print(
            "Could not fined {0} in defined basenames {1}".format(
                tokens[0], defined_baseconfigs
            )
        )
        return None
    else:
        config = (name, "", defined_baseconfigs[tokens[0]])
    for token in tokens[1:]:
        if token not in defined_xtracfgs:
            print(
                "Could not find {0} in defined xtraconfigs {1}".format(
                    token, defined_xtracfgs
                )
            )
            return None
        else:
            oldName, oldXtra, oldBasename = config
            config = (
                oldName,
                oldXtra + "\n#{0}\n{1}\n".format(token, defined_xtracfgs[token]),
                oldBasename,
            )
    return config


def load_defined_yamls():
    define_yamls = glob.glob(os.path.join(this_directory, "apps/define-*.yml"))
    for def_yaml in define_yamls:
        parse_app_definition_yaml(
            os.path.join(this_directory, "apps", def_yaml), defined_apps
        )
    define_yamls = glob.glob(os.path.join(this_directory, "configs/define-*.yml"))
    for def_yaml in define_yamls:
        parse_config_definition_yaml(
            os.path.join(this_directory, "configs", def_yaml),
            defined_baseconfigs,
            defined_xtracfgs,
        )


def parse_app_definition_yaml(def_yml, apps):
    benchmark_yaml = yaml.load(open(def_yml), Loader=yaml.FullLoader)
    for suite in benchmark_yaml:
        apps[suite] = []
        for exe in benchmark_yaml[suite]["execs"]:
            exe_name = list(exe.keys())[0]
            args_list = list(exe.values())[0]
            count = 0
            for runparms in args_list:
                args = runparms["args"]
                if "accel-sim-mem" not in runparms:
                    runparms["accel-sim-mem"] = "4G"
                apps[suite + ":" + exe_name + ":" + str(count)] = []
                apps[suite + ":" + exe_name + ":" + str(count)].append(
                    (
                        benchmark_yaml[suite]["exec_dir"],
                        benchmark_yaml[suite]["data_dirs"],
                        exe_name,
                        [args],
                    )
                )
                count += 1
            apps[suite].append(
                (
                    benchmark_yaml[suite]["exec_dir"],
                    benchmark_yaml[suite]["data_dirs"],
                    exe_name,
                    args_list,
                )
            )
            apps[suite + ":" + exe_name] = []
            apps[suite + ":" + exe_name].append(
                (
                    benchmark_yaml[suite]["exec_dir"],
                    benchmark_yaml[suite]["data_dirs"],
                    exe_name,
                    args_list,
                )
            )
    return


def parse_config_definition_yaml(def_yml, defined_baseconfigs, defined_xtracfgs):
    configs_yaml = yaml.load(open(def_yml), Loader=yaml.FullLoader)
    for config in configs_yaml:
        if "base_file" in configs_yaml[config]:
            defined_baseconfigs[config] = os.path.expandvars(
                configs_yaml[config]["base_file"]
            )
        elif "extra_params" in configs_yaml[config]:
            defined_xtracfgs[config] = configs_yaml[config]["extra_params"]
    return


def gen_apps_from_suite_list(app_list):
    benchmarks = []
    for app in app_list:
        benchmarks += defined_apps[app]
    return benchmarks


def gen_configs_from_list(cfg_list):
    configs = []
    for cfg in cfg_list:
        configs.append(get_config(cfg, defined_baseconfigs, defined_xtracfgs))
    return configs


def get_cuda_version(this_directory):
    # Get CUDA version
    nvcc_out_filename = os.path.join(
        this_directory, "nvcc_out.{0}.txt".format(os.getpid())
    )
    nvcc_out_file = open(nvcc_out_filename, "w+")
    subprocess.call(["nvcc", "--version"], stdout=nvcc_out_file)
    nvcc_out_file.seek(0)
    cuda_version = re.sub(
        r".*release (\d+\.\d+).*",
        r"\1",
        nvcc_out_file.read().strip().replace("\n", " "),
    )
    nvcc_out_file.close()
    os.remove(nvcc_out_filename)
    os.environ["CUDA_VERSION"] = cuda_version
    return cuda_version


class PathMissing(Exception):
    pass


# This function exists so that this file can accept both absolute and relative paths
# If no name is provided it sets the default
# Either way it does a test if the absolute path exists and if not, tries a relative path
def file_option_test(name, default, this_directory):
    name = os.path.expandvars(name)
    if name == "":
        if default == "":
            return ""
        else:
            name = os.path.join(this_directory, default)
    try:
        with open(name):
            pass
    except IOError:
        name = os.path.join(os.getcwd(), name)
        try:
            with open(name):
                pass
        except IOError:
            raise PathMissing("Error - cannot open file {0}".format(name))
    return name


def dir_option_test(name, default, this_directory):
    name = os.path.expandvars(name)
    if name == "":
        name = os.path.join(this_directory, default)
    if not os.path.isdir(name):
        name = os.path.join(os.getcwd(), name)
        if not os.path.isdir(name):
            raise PathMissing("Error - directory test fails for {0}".format(name))
    return name


def parse_run_simulations_options():
    parser = OptionParser()
    parser.add_option(
        "-B",
        "--benchmark_list",
        dest="benchmark_list",
        help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for "
        + "the benchmark suite names.",
        default="rodinia_2.0-ft",
    )
    parser.add_option(
        "-C",
        "--configs_list",
        dest="configs_list",
        help="a comma seperated list of configs to run. See configs/define-*.yml for "
        + "the config names.",
        default="QV100",
    )
    parser.add_option(
        "-p",
        "--benchmark_exec_prefix",
        dest="benchmark_exec_prefix",
        help="When submitting the job to torque this string"
        + " is placed before the command line that runs the benchmark. "
        + " Useful when wanting to run valgrind.",
        default="",
    )
    parser.add_option(
        "-r",
        "--run_directory",
        dest="run_directory",
        help="Name of directory in which to run simulations",
        default="",
    )
    parser.add_option(
        "-n",
        "--no_launch",
        dest="no_launch",
        action="store_true",
        help="When set, no torque jobs are launched.  However, all"
        + " the setup for running is performed. ie, the run"
        + " directories are created and are ready to run."
        + " This can be useful when you want to create a new"
        + " configuration, but want to test it locally before "
        + " launching a bunch of jobs.",
    )
    parser.add_option(
        "-s",
        "--simulator_dir",
        dest="simulator_dir",
        help="Point this to the directory that your .so is stored in. If nothing is input here - "
        + "the scripts will assume that you are using the so built in GPGPUSIM_ROOT or ACCELSIM_ROOT.",
        default="",
    )
    parser.add_option(
        "-N",
        "--launch_name",
        dest="launch_name",
        default="",
        help="Pass if you want to name the launch. This will determine the name of the logfile.\n"
        + "If you do not name the file, it will just use the current date/time.",
    )
    parser.add_option(
        "-T",
        "--trace_dir",
        dest="trace_dir",
        default="",
        help="Pass this option to run the simulator in trace-driven mode."
        + " The directory passed should be the root of all the trace files.",
    )
    parser.add_option(
        "-M",
        "--job_mem",
        dest="job_mem",
        default=None,
        help="Memory usgae of the job be sure to specify the units i.e. 4G, 900M, etc..",
    )
    parser.add_option(
        "-l",
        "--launcher",
        dest="launcher",
        default="",
        help="Specify how jobs will be launched. Select one of sbatch (slurm), qsub (torque), "
        "local. By default, we test for slurm, then torque, then just use local if "
        "you have neither.",
    )
    parser.add_option(
        "-c",
        "--cores",
        dest="cores",
        default=None,
        help="Specify the core limit when using procman. If nothing is specified, all the cores"
        " on the local node will be used.",
    )
    parser.add_option(
        "-a",
        "--accelwattch_HW",
        dest="accelwattch_HW",
        action="store_true",
        help="Enable passing hw_perf_bench_name for accelwattch hw and hybrid runs to config file.",
    )

    (options, args) = parser.parse_args()
    # Parser seems to leave some whitespace on the options, getting rid of it
    if options.trace_dir != "":
        options.trace_dir = dir_option_test(
            options.trace_dir.strip(), "", this_directory
        )
    options.configs_list = options.configs_list.strip()
    options.benchmark_exec_prefix = options.benchmark_exec_prefix.strip()
    options.benchmark_list = options.benchmark_list.strip()
    options.run_directory = options.run_directory.strip()
    options.simulator_dir = options.simulator_dir.strip()
    options.launch_name = options.launch_name.strip()
    if options.job_mem != None:
        options.job_mem = options.job_mem.strip()
    return (options, args)


# After collection, spew out the tables
def print_stat(
    stat_name,
    all_named_kernels,
    apps_and_args,
    configs,
    stat_map,
    cfg_as_rows,
    do_averages,
):
    csv_str = ""
    DIVISION = "-" * 100
    if cfg_as_rows:
        num_commas = len(apps_and_args)
    else:
        num_commas = len(configs)
    if do_averages:
        num_commas += 1
    csv_str += DIVISION + ("," * num_commas) + "\n"

    running_total = 0
    total_num = 0
    if cfg_as_rows:
        csv_str += stat_name + ("," * num_commas) + "\nCFG,"
        for appargs in apps_and_args:
            knames = all_named_kernels[appargs]
            for kname in knames:
                if kname == "":
                    continue
                csv_str += appargs + "--" + kname + ","
        if do_averages:
            csv_str += "AVG,"

        csv_str = csv_str[:-1]
        csv_str += "\n"
        for config in configs:
            csv_str += config + ","
            for appargs in apps_and_args:
                knames = all_named_kernels[appargs]
                for kname in knames:
                    if kname == "":
                        continue
                    if kname + appargs + config + stat_name in stat_map:
                        csv_str += (
                            str(stat_map[kname + appargs + config + stat_name]) + ","
                        )
                        try:
                            running_total += float(
                                stat_map[kname + appargs + config + stat_name]
                            )
                            total_num += 1
                        except:
                            pass
                    else:
                        csv_str += "NA,"
            if do_averages:
                if total_num != 0:
                    csv_str += "{0:.1f},".format(running_total / total_num)
                else:
                    csv_str += "NA,"
            running_total = 0
            total_num = 0
            csv_str = csv_str[:-1]
            csv_str += "\n"

    else:
        csv_str += stat_name + ("," * num_commas) + "\nAPPS,"
        for config in configs:
            csv_str += config + ","

        if do_averages:
            csv_str += "AVG,"
        csv_str = csv_str[:-1]
        csv_str += "\n"
        for appargs in apps_and_args:
            knames = all_named_kernels[appargs]
            for kname in knames:
                if kname == "":
                    continue
                csv_str += appargs + "--" + kname + ","
                for config in configs:
                    if kname + appargs + config + stat_name in stat_map:
                        csv_str += (
                            str(stat_map[kname + appargs + config + stat_name]) + ","
                        )
                        try:
                            running_total += float(
                                stat_map[kname + appargs + config + stat_name]
                            )
                            total_num += 1
                        except:
                            pass
                    else:
                        csv_str += "NA,"

                if do_averages:
                    if total_num != 0:
                        csv_str += "{0:.1f},".format(running_total / total_num)
                    else:
                        csv_str += "NA,"
                running_total = 0
                total_num = 0
                csv_str = csv_str[:-1]
                csv_str += "\n"

    csv_str = csv_str[:-1]
    csv_str += "\n"
    print(csv_str)
