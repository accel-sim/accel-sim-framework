#!/usr/bin/env python3

from __future__ import print_function
from optparse import OptionParser
import re
import os
import subprocess
import sys
import common
import math
import yaml
import time

millnames = ['',' K',' M',' B',' T']
def millify(n):
    n = float(n)
    if math.isnan(n):
        return "NaN"
    if math.isinf(n):
        return "inf"
    millidx = max(0,min(len(millnames)-1,
                    int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return '{:.3f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

#*********************************************************--
# main script start
#*********************************************************--
start_time = time.time()
files_parsed = 0
bytes_parsed = 0

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

help_str = "There are 3 ways to use this file, select only one of the following:" +\
           " \n1) Specify a sim_name: \"-N <the-name>\"\n"+\
           "    If you do this, then the job launching logfiles"+\
           " will be searched for the most recent sim_name job launch."+\
           " Those specific output files will be parsed."+\
           " \n2) Specify a logfile: \"-l <the-file>\"\n"+\
           "    If you do this, then the jobs in the specific logfile will be parsed" +\
           " If no options are specified, then it basically defaults to the -l"+\
           " option using the latest logfile." +\
           " \n3) Specify a list of configs -C and benchmarks -B suite names you want data for."


parser = OptionParser(usage=help_str)
parser.add_option("-l", "--logfile", dest="logfile",
                  help="The logfile the status is based on. "+\
                        "By default, we will base it on the latest simulations launched.\n" +\
                        "specify \"all\" to use all the simulation logfiles in the directory",
                  default="")
parser.add_option("-r", "--run_dir", dest="run_dir",
                  help="The directory where the benchmark/config directories exist.", default="")
parser.add_option("-N", "--sim_name", dest="sim_name",
                  help="If you are launching run_simulations.py with the \"-N\" option" +\
                       " then you can run ./job_status.py with \"-N\" and it will" + \
                       " give you the status of the latest run with that name."+ \
                       " if you want older runs from this name, then just point it directly at the"+\
                       " logfile with \"-l\"", default="")
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                  help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for " +\
                        "the benchmark suite names.",
                  default="")
parser.add_option("-C", "--configs_list", dest="configs_list",
                  help="a comma seperated list of configs to run. See configs/define-*.yml for " +\
                        "the config names.",
                  default="")
#parser.add_option("-b", "--simulator_build", dest="simulator_build", default="",
#                  help="If you only want data from a particular build of the simulator, specify this flag.")
parser.add_option("-s", "--stats_yml", dest="stats_yml", default="",
                  help="The yaml file that defines the stats you want to collect."+\
                       " by default it uses stats/example_stats.yml")
parser.add_option("-k", "--per_kernel", dest="per_kernel", action="store_true",
                  help="Aggregate the statistics for each named kernel")
parser.add_option("-K", "--kernel_instance", dest="kernel_instance", action="store_true",
                  help="Print stats for each individual kernel the statistics for each named kernel")
parser.add_option("-R", "--configs_as_rows", dest="configs_as_rows", action="store_true",
                  help="instread of apps as rows in the csv, make configs as rows.")
parser.add_option("-I", "--ignore_failures", dest="ignore_failures", action="store_true",
                  help="If an app crashed, still collect its data")
parser.add_option("-A", "--do_averages", dest="do_averages", action="store_true",
                  help="Print the averages for each statistic")
(options, args) = parser.parse_args()
options.logfile = options.logfile.strip()
options.run_dir = options.run_dir.strip()
options.sim_name = options.sim_name.strip()

common.load_defined_yamls()


cuda_version = common.get_cuda_version( this_directory )
options.run_dir = common.dir_option_test( options.run_dir, this_directory + ("../../sim_run_%s/"%cuda_version),
                                          this_directory )
if not os.path.isdir(options.run_dir):
    exit(options.run_dir + " does not exist - specify the run directory where the benchmark/config dirs exist")

options.stats_yml = common.file_option_test( options.stats_yml, os.path.join( this_directory, "stats", "example_stats.yml" ),
                                            this_directory )
stat_map = {}
configs = []
apps_and_args = []
exes_and_args = []
specific_jobIds = {}

stats_to_pull = {}
stats_yaml = yaml.load(open(options.stats_yml), Loader=yaml.FullLoader )
stats= {}
for stat in stats_yaml['collect_aggregate']:
    stats_to_pull[stat] = re.compile(stat), "agg"

for stat in stats_yaml['collect_abs']:
    stats_to_pull[stat] = re.compile(stat), "abs"

for stat in stats_yaml['collect_rates']:
    stats_to_pull[stat] = re.compile(stat), "rate"


if options.configs_list != "" and options.benchmark_list != "":
    for app in common.gen_apps_from_suite_list(options.benchmark_list.split(",")):
        a,b,exe_name,args_list = app
        for argpair in args_list:
            args = argpair["args"]
            apps_and_args.append( os.path.join(exe_name, common.get_argfoldername(args) ) )
    for config, params, gpuconf_file in common.gen_configs_from_list( options.configs_list.split(",") ):
        configs.append( config )
else:
    # This code gets the logfiles to pull the stats from if you are using the "-l" or "-N" option
    parsed_logfiles = []
    logfiles_directory = this_directory + "../job_launching/logfiles/"
    if options.logfile == "":
        if not os.path.exists(logfiles_directory):
            exit("No logfile specified and the default logfile directory cannot be found")
        all_logfiles = [os.path.join(logfiles_directory, f) \
                           for f in os.listdir(logfiles_directory) if(re.match(r'sim_log.*',f))]
        if len(all_logfiles) == 0:
            exit("ERROR - No Logfiles in " + logfiles_directory)
        if options.sim_name != "":
            named_sim = []
            for logf in all_logfiles:
                match_str = r".*\/sim_log\.{0}\..*".format( options.sim_name )
                if re.match( match_str, logf ):
                    named_sim.append( logf )
            if len( named_sim ) == 0:
                exit( "Could not find logfiles for job with the name \"{0}\"".format( options.sim_name ) )
            all_logfiles = named_sim
        parsed_logfiles.append(max(all_logfiles, key=os.path.getmtime))
    elif options.logfile == "all":
        parsed_logfiles = [os.path.join(logfiles_directory, f) \
                        for f in os.listdir(logfiles_directory) if(re.match(r'sim_log.*\.latest',f))]
    else:
        parsed_logfiles.append(common.file_option_test( options.logfile, "", this_directory ))

    print("Using logfiles " + str(parsed_logfiles), file=sys.stderr)

    for logfile in parsed_logfiles:
        if not os.path.isfile(logfile):
            exit("Cannot open Logfile " + logfile)

        with open( logfile ) as f:
            added_cfgs = set()
            added_apps = set()
            for line in f:
                jobtime, jobId, app ,args, config, jobname = line.split()
                if config not in added_cfgs:
                    configs.append(config)
                    added_cfgs.add(config)
                app_and_args = os.path.join( app.replace('/','_'), args )
                if app_and_args not in added_apps:
                    apps_and_args.append( app_and_args )
                    exe_and_args = os.path.join( os.path.basename(app), args)
                    exes_and_args.append(exe_and_args)
                    added_apps.add(app_and_args)
                specific_jobIds[ config + app_and_args ] = (jobId,jobname)

all_named_kernels = {}
for idx, app_and_args in enumerate(apps_and_args):
    all_named_kernels[app_and_args] = []
    for config in configs:
        # now get the right output file
        output_dir = os.path.join(options.run_dir, app_and_args, config)
        if not os.path.isdir( output_dir ):
            print("WARNING the outputdir " + output_dir + " does not exist", file=sys.stderr)
            continue

        if config + app_and_args in specific_jobIds:
            jobId,jobname = specific_jobIds[ config + app_and_args ]
            torque_submname = re.sub(r".*\.([^\s]*-commit-.*)", r"\1", jobname)
            outfile = os.path.join(output_dir, exes_and_args[idx].replace("/", "-") + "." +\
               torque_submname + "." + "o" + jobId)
        else:
            all_outfiles = [os.path.join(output_dir, f) \
                           for f in os.listdir(output_dir) if(re.match(r'.*\.o[0-9]+',f))]
            if len(all_outfiles) != 0:
                outfile = max(all_outfiles, key=os.path.getmtime)
            else:
                continue

        stat_found = set()

        if not os.path.isfile( outfile ):
            print("WARNING - " + outfile + " does not exist", file=sys.stderr)
            continue

        # Do a quick 100-line pass to get the GPGPU-Sim Version number
        MAX_LINES = 100
        count = 0
        f = open(outfile)
        for line in f:
            count += 1
            if count >= MAX_LINES:
                break
            gpgpu_build_match = re.match(".*GPGPU-Sim.*\[build\s+(.*)\].*", line)
            if gpgpu_build_match:
                stat_map["all_kernels" + app_and_args + config + "GPGPU-Sim-build"] = gpgpu_build_match.group(1)
                break
            accelsim_build_match = re.match("Accel-Sim.*\[build\s+(.*)\].*", line)
            if accelsim_build_match:
                stat_map["all_kernels" + app_and_args + config + "Accel-Sim-build"] = accelsim_build_match.group(1)
        f.close()

        # Do a quick 10000-line reverse pass to make sure the simualtion thread finished
        SIM_EXIT_STRING = "GPGPU-Sim: \*\*\* exit detected \*\*\*"
        exit_success = False
        MAX_LINES = 10000
        BYTES_TO_READ = int(250 * 1024 * 1024)
        count = 0
        f = open(outfile)
        fsize = int(os.stat(outfile).st_size)
        if fsize > BYTES_TO_READ:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - BYTES_TO_READ, os.SEEK_SET)
        lines = f.readlines()
        for line in reversed(lines):
            count += 1
            if count >= MAX_LINES:
                break
            exit_match = re.match(SIM_EXIT_STRING, line)
            if exit_match:
                exit_success = True
                break
        del lines
        f.close()

        if not exit_success:
            print("WARNING - Detected that {0} does not contain a terminating string from GPGPU-Sim. The output is potentially invalid".format(outfile),
                file=sys.stderr)
            if not options.ignore_failures:
                continue

        if not options.per_kernel:
            if len(all_named_kernels[app_and_args]) == 0:
                all_named_kernels[app_and_args].append("final_kernel")
            BYTES_TO_READ = int(250 * 1024 * 1024)
            count = 0
            f = open(outfile)
            fsize = int(os.stat(outfile).st_size)
            files_parsed += 1
            if fsize > BYTES_TO_READ:
                f.seek(0, os.SEEK_END)
                f.seek(f.tell() - BYTES_TO_READ, os.SEEK_SET)
                bytes_parsed += BYTES_TO_READ
            else:
                bytes_parsed += fsize
            lines = f.readlines()
            for line in reversed(lines):
                # pull out some stats
                for stat_name, tup in stats_to_pull.items():
                    token, statType = tup
                    if stat_name in stat_found:
                        continue
                    existance_test = token.search( line.rstrip() )
                    if existance_test != None:
                        stat_found.add(stat_name)
                        number = existance_test.group(1).strip()
                        stat_map["final_kernel" + app_and_args + config + stat_name] = number
                if len(stat_found) == len(stats_to_pull):
                    break
            del lines
            f.close()
        else:
            current_kernel =""
            last_kernel = ""
            raw_last = {}
            running_kcount = {}
            files_parsed += 1
            bytes_parsed += os.stat(outfile).st_size
            f = open(outfile)
            #print("Parsing File {0}. Size: {1}".format(outfile, millify(os.stat(outfile).st_size)))
            for line in f:
                # If we ended simulation due to too many insn - ignore the last kernel launch, as it is no complete.
                # Note: This only appies if we are doing kernel-by-kernel stats
                last_kernel_break = re.match("GPGPU-Sim: \*\* break due to reaching the maximum cycles \(or instructions\) \*\*", line)
                if last_kernel_break:
                    print("NOTE::::: Found Max Insn reached in {0} - ignoring last kernel.".format(outfile), file=sys.stderr)
                    for stat_name in stats_to_pull.keys():
                        if current_kernel + app_and_args + config + stat_name in stat_map:
                            del stat_map[current_kernel + app_and_args + config + stat_name]

                kernel_match = re.match("kernel_name\s+=\s+(.*)", line);
                if kernel_match:
                    last_kernel = current_kernel
                    current_kernel = kernel_match.group(1).strip()

                    if options.kernel_instance:
                        if current_kernel not in running_kcount:
                            running_kcount[current_kernel] = 0
                        else:
                            running_kcount[current_kernel] += 1
                        current_kernel += "--" + str(running_kcount[current_kernel])

                    if current_kernel not in all_named_kernels[app_and_args]:
                        all_named_kernels[app_and_args].append(current_kernel)

                    if current_kernel + app_and_args + config + "k-count" in stat_map:
                        stat_map[current_kernel + app_and_args + config + "k-count"] += 1
                    else:
                        stat_map[current_kernel + app_and_args + config + "k-count"] = 1
                    continue

                for stat_name, tup in stats_to_pull.items():
                    token, statType = tup
                    existance_test = token.search( line.rstrip() )
                    if existance_test != None:
                        stat_found.add(stat_name)
                        number = existance_test.group(1).strip()
                        if statType != "agg":
                            stat_map[current_kernel + app_and_args + config + stat_name] = number
                        elif current_kernel + app_and_args + config + stat_name in stat_map:
                            if stat_name in raw_last:
                                stat_last_kernel = raw_last[stat_name]
                            else:
                                stat_last_kernel = 0.0
                            raw_last[ stat_name ] = float(number)
                            stat_map[current_kernel + app_and_args + config + stat_name] += ( float(number) - stat_last_kernel )
                        else:
                            if last_kernel + app_and_args + config + stat_name in stat_map:
                                stat_last_kernel = raw_last[stat_name]
                            else:
                                stat_last_kernel = 0.0
                            raw_last[stat_name] = float(number)
                            stat_map[current_kernel + app_and_args + config + stat_name] = ( float(number) - stat_last_kernel )
# Just adding this in here since it is a special case and is not parsed like everything else, because you need
# to read from the beginning not the end
#if options.per_kernel and not options.kernel_instance:
#    stats_yaml['collect'].append("k-count")


# Print any stats that do not make sense on a per-kernel basis ever (like GPGPU-Sim Build)
all_kernels = {}
for appargs in apps_and_args:
    all_kernels[appargs] = ["all_kernels"]

#stat_name, all_named_kernels, apps_and_args, configs, stat_map, cfg_as_rows
common.print_stat( "Accel-Sim-build", all_kernels, apps_and_args, configs, stat_map, options.configs_as_rows, options.do_averages )
common.print_stat( "GPGPU-Sim-build", all_kernels, apps_and_args, configs, stat_map, options.configs_as_rows, options.do_averages )

for stat_name in ( stats_yaml['collect_aggregate'] +\
                   stats_yaml['collect_abs'] +\
                   stats_yaml['collect_rates'] ):
    common.print_stat( stat_name, all_named_kernels, apps_and_args, configs, stat_map, options.configs_as_rows, options.do_averages )

duration = time.time() - start_time

print("Script exec time {0:.2f} seconds. {1} files and {2}B parsed. {3}B/s".\
    format(duration , files_parsed, millify(bytes_parsed),
    millify(float(bytes_parsed)/float(duration))), file=sys.stderr)
