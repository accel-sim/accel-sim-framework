#!/usr/bin/env python

from optparse import OptionParser
import re
import os
import subprocess
import sys
import common
import math
import yaml

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

#*********************************************************--
# main script start
#*********************************************************--
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

help_str = "There are 3 ways to use this file" +\
           " 1) Specify a sim_name: \"-N <the-name>\"\n"+\
           "    If you do this, then the job launching logfiles"+\
           " will be searched for the most recent sim_name job launch."+\
           " Those specific output files will be parsed."+\
           " 2) Specify a logfile: \"-l <the-file>\"\n"+\
           "    If you do this, then the jobs in the specific logfile will be parsed" +\
           " If no options are specified, then it basically defaults to the -l"+\
           " option using the latest logfile."
#           " 3) Specify a configs yml: \"-y <the-yaml>\"\n"+\
#           "    The \"configs\" and \"benchmark\" lists in the yaml are parsed."+\
#           " The most recent runs of these bench+config options are searched.\n"+\


parser = OptionParser(usage=help_str)
parser = OptionParser()
parser.add_option("-l", "--logfile", dest="logfile",
                  help="The logfile the status is based on. "+\
                        "By default, we will base it on the latest simulations launched.\n" +\
                        "specify \"all\" to use all the simulation logfiles in the directory",
                  default="")
parser.add_option("-r", "--run_dir", dest="run_dir",
                  help="The directory where the benchmark/config directories exist.", default="")
parser.add_option("-N", "--sim_name", dest="sim_name",
                  help="If you are launchign run_simulations.py with the \"-N\" option" +\
                       " then you can run ./job_status.py with \"-N\" and it will" + \
                       " give you the status of the latest run with that name."+ \
                       " if you want older runs from this name, then just point it directly at the"+\
                       " logfile with \"-l\"", default="")
#parser.add_option("-y", "--configs_yml", dest="configs_yml", default="",
#                  help="If this option is specified, then sim_name and logfile are ignored." +\
#                       "Instead, the output files that will be parsed will")
#parser.add_option("-b", "--simulator_build", dest="simulator_build", default="",
#                  help="If you only want data from a particular build of the simulator, specify this flag.")
parser.add_option("-s", "--stats_yml", dest="stats_yml", default="",
                  help="The yaml file that defines the stats you want to collect."+\
                       " by default it uses stats/example_stats.yml")
(options, args) = parser.parse_args()
options.logfile = options.logfile.strip()
options.run_dir = options.run_dir.strip()
options.sim_name = options.sim_name.strip()

cuda_version = common.get_cuda_version()

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

print "Using logfiles " + str(parsed_logfiles)

options.run_dir = common.dir_option_test( options.run_dir, this_directory + ("../../sim_run_%s/"%cuda_version),
                                          this_directory )
options.stats_yml = common.file_option_test( options.stats_yml, os.path.join( this_directory, "stats", "example_stats.yml" ),
                                            this_directory )

stats_to_pull = {}
stats_yaml = yaml.load(open(options.stats_yml))
stats= {}
for stat in stats_yaml['collect']:
    stats_to_pull[stat] = re.compile(stats_yaml['collect'][stat])#.decode('string_escape')

if not os.path.isdir(options.run_dir):
    exit(options.run_dir + " does not exist - specify the run directory where the benchmark/config dirs exist")

stat_map = {}
configs = set()
benchmark_args = set()

# At this point we have the logfile we want to get a synopsis for.
for logfile in parsed_logfiles:
    if not os.path.isfile(logfile):
        exit("Cannot open Logfile " + logfile)

    # Create the output file
    base_logname = os.path.basename(logfile)

    # Parse the logfile for job ids
    with open( logfile ) as f:
        for line in f:
            time, jobId, app ,args, config, jobname = line.split()
            configs.add(config)
            benchmark_args.add(app + args)

            # now get the right logfile
            output_dir = os.path.join(options.run_dir, app, args, config)
            if not os.path.isdir( output_dir ):
                print("WARNING the outputdir " + output_dir + " does not exist")
                continue

            outfile = os.path.join(output_dir, app + "-" + args + "." + "o" + jobId)

            stat_found = set()

            if not os.path.isfile( outfile ):
                print "WARNING - " + outfile + " does not exist"
                continue

            # Only go up for 1000 lines looking for stuff
            MAX_LINES = 100000
            count = 0
            for line in reversed(open(outfile).readlines()):
                count += 1
                if count >= MAX_LINES:
                    break

                # pull out some stats
                for stat_name, token in stats_to_pull.iteritems():
                    if stat_name in stat_found:
                        continue
                    existance_test = token.search( line.rstrip() )
                    if existance_test != None:
                        stat_found.add(stat_name)
                        number = existance_test.group(1).strip()
                        stat_map[app + args + config + stat_name] = number
                if len(stat_found) == len(stats_to_pull):
                    break

# After collection, spew out the tables
DIVISION = "-" * 100
csv_str = ""
for stat_name, token in stats_to_pull.iteritems():
    csv_str += DIVISION + "\n"
    csv_str += stat_name + "\n,"
    for config in configs:
        csv_str += config + ","
    csv_str += "\n"
    for appargs in benchmark_args:
        csv_str += appargs + ","
        for config in configs:
            if appargs + config + stat_name in stat_map:
                csv_str += stat_map[appargs + config + stat_name] + ","
            else:
                csv_str += "NA,"
        csv_str += "\n"
    csv_str += "\n"

print csv_str
