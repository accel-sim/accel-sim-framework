#!/usr/bin/env python

from optparse import OptionParser
import re
import os
import subprocess
import sys
import common
import math

# uses torque's tracejob to determine the job status
def get_job_status( jobId ):
    job_status = { "state" : "WAITING_TO_RUN",
                   "exec_host" : "UNKNOWN" }
    trace_out_filename = "/tmp/trace_out.txt"
    trace_out_file = open(trace_out_filename, 'w+')
    if subprocess.call(["qstat" ,"-f", jobId],
        stdout=trace_out_file, stderr=trace_out_file) < 0:
        exit("Error Launching Tracejob Job")
    else:
        # Parse the torque output for just the numeric ID
        trace_out_file.seek(0)
        trace_out = re.sub( "\n", " ", trace_out_file.read().strip() )
        state_match = re.search( "job_state\s=\s([^\s]*)", trace_out )
        if state_match != None:
            if (state_match.group(1) == 'R' or state_match.group(1) == 'E'):
                job_status[ "state" ] = "RUNNING"
            elif state_match.group(1) == 'C':
                job_status[ "state" ] = "COMPLETE_NO_OTHER_INFO"
            host_match = re.search( "exec_host\s=\s([^\s]*)", trace_out )
            if host_match != None:
                job_status[ "exec_host" ] = host_match.group(1)
        trace_out_file.close()
        os.remove(trace_out_filename)
    return job_status

def isNumber( s ):
    try:
        int (s)
        return True
    except ValueError:
        try:
            float( s )
            return True
        except ValueError:
            return False

millnames = ['',' K',' M',' B',' T']
def millify(n):
    n = float(n)
    if math.isnan(n):
        return "NaN"
    millidx = max(0,min(len(millnames)-1,
                    int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return '{:.3f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

#*********************************************************--
# main script start
#*********************************************************--
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

parser = OptionParser()
parser.add_option("-l", "--logfile", dest="logfile",
                  help="The logfile the status is based on. "+\
                        "By default, we will base it on the latest simulations launched.\n" +\
                        "specify \"all\" to use all the simulation logfiles in the directory",
                  default="")
parser.add_option("-n", "--num_lines", dest="num_lines",
                  help="Number of lines of the *.o file to print when an error occurs." +\
                        " Default is 10", default="10")
parser.add_option("-r", "--run_dir", dest="run_dir",
                  help="The directory where the benchmark/config directories exist.", default="")
parser.add_option("-N", "--sim_name", dest="sim_name",
                  help="If you are launchign run_simulations.py with the \"-N\" option" +\
                       " then you can run ./job_status.py with \"-N\" and it will" + \
                       " give you the status of the latest run with that name."+ \
                       " if you want older runs from this name, then just point it directly at the"+\
                       " logfile with \"-l\"", default="")
(options, args) = parser.parse_args()
options.logfile = options.logfile.strip()
options.num_lines = options.num_lines.strip()
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

if not os.path.isdir(options.run_dir):
    exit(options.run_dir + " does not exist - specify the run directory where the benchmark/config dirs exist")

# Searches the output file for these strings.
# If they exist, it assumes that the functional
# Test implemented in the CPU program has passed
# Searches the output file and the stderr file for these strings
# If they exists, then something has gone wrong.
status_strings = { "passed" : "FUNC_TEST_PASSED",
                   "PASSED" : "FUNC_TEST_PASSED",
                   "Assertion" : "ASSERT",
                   "Segmentation fault" : "SEGF",
                   "Aborted" : "ABORTED",
                   "PBS" : "TORQUE_KILLED",
                   "deadlock.*detected" : "DEADLOCK",
                   "failed to open display" : "FAIL_NO_DISPLAY",
                   "FAILED" : "FUNC_TEST_FAILED",
                   "failed" : "FUNC_TEST_FAILED",
                   "Failed" : "FUNC_TEST_FAILED",
                   "XML Parsing error.*gpuwattch" : "NO_GPU_WATTCH_CFG" }

# Also searches the output files for these stats and prints them in the log
stats_to_pull = { "SIM_TIME": "gpgpu_simulation_time\s*=[^1-9]*(.*)",
                  "TOT_INSN" : "gpu_tot_sim_insn\s*=\s*(.*)",
                  "TOT_IPC" : "gpu_tot_ipc\s*=\s*(.*)" }

ROW_STRING = "{jobId:<10.10}\t{exec_node:<30.30}\t{app:<20.20}\t{args:<20.20}\t" +\
             "{gpusim_version:20.20}\t{config:20.20}\t{status:40.40}\t{stat:50}"

# At this point we have the logfile we want to get a synopsis for.
for logfile in parsed_logfiles:
    if not os.path.isfile(logfile):
        exit("Cannot open Logfile " + logfile)


    # Create the output file
    base_logname = os.path.basename(logfile)

    # Parse the logfile for job ids
    with open( logfile ) as f:
        header = ROW_STRING.format( jobId="TorqueJob",exec_node="Node",app="App",args="AppArgs",
                gpusim_version="GPGPU-SimVersion",config="GPGPU-SimConfig",
                status="JobStatus", stat="Basic GPGPU-Sim Stats" )
        print header
        print "-" * len(header)

        failed_job_text = ""
        failed_jobs_summary = ""
        num_jobs = 0
        num_passed = 0
        a_job_failed = False
        for line in f:
            time, jobId, app ,args, config, jobname = line.split()

            # now get the right logfile
            output_dir = os.path.join(options.run_dir, app, args, config)
            if not os.path.isdir( output_dir ):
                print("WARNING the outputdir " + output_dir + " does not exist")
                continue

            num_jobs += 1
            errfile = os.path.join(output_dir, app + "-" + args + "." + "e" + jobId)
            outfile = os.path.join(output_dir, app + "-" + args + "." + "o" + jobId)

            status_string = ""
            additional_stats = ""
            stat_found = set()
            status_found = set()

            job_status = get_job_status( jobId )
            if ( job_status[ "state" ] == "WAITING_TO_RUN" or job_status[ "state" ] == "RUNNING" ) \
                and not os.path.isfile( outfile ):
                files_to_check = []
                status_string = job_status[ "state" ]
            else:
                files_to_check = [ outfile, errfile ]
                status_string = "COMPLETE_NO_OTHER_INFO"

            exec_node = job_status[ "exec_host" ]

            for sim_file in files_to_check:
                if not os.path.isfile( sim_file ):
                    print "WARNING - " + sim_file + " does not exist"
                    continue

                # Only go up for 1000 lines looking for stuff
                MAX_LINES = 100000
                count = 0
                for line in reversed(open(sim_file).readlines()):
                    count += 1
                    if count >= MAX_LINES:
                        break
                    # search for the failue conditions
                    for token, name in status_strings.iteritems():
                        if name in status_found:
                            continue
                        if re.search( token, line.rstrip() ):
                            failed = True
                            status_found.add( name )

                    # pull out some stats
                    for name,token in stats_to_pull.iteritems():
                        if token in stat_found:
                            continue
                        existance_test = re.search( token, line.rstrip() )
                        if existance_test != None:
                            stat_found.add(token)
                            number = existance_test.group(1).strip()
                            if isNumber( number ):
                                number = millify( number )
                            if additional_stats != "":
                                additional_stats += "\t{0}={1}".format( name, number )
                            else:
                                additional_stats = "{0}={1}".format( name, number )
                    if len(stat_found) == len(stats_to_pull):
                        break

            if len( status_found ) > 0:
                status_string = ", ".join( status_found )
            elif os.path.exists( errfile ) and os.stat( errfile ).st_size > 0:
                status_string = "COMPLETE_ERR_FILE_HAS_CONTENTS"

            gpgpu_git_commit = re.sub(r".*-commit-([^_]{7})[^_]+_(.*)\.so", r"\1-\2", jobname)
            job_summary = ROW_STRING.format( jobId=jobId, exec_node=exec_node, app=app, args=args,
                            config=config, status=status_string, stat=additional_stats,
                            gpusim_version=gpgpu_git_commit )
            print job_summary

            if "FUNC_TEST_PASSED" == status_string:
                num_passed += 1
            elif ( "WAITING_TO_RUN" not in status_string \
                and "RUNNING" not in status_string ):
                failed_jobs_summary += job_summary + "\n"
                failed_job_text += "**********************************************************\n"
                failed_job_text += "{0}-{1}--{2}. Status={3}\n".format( app, args, config, status_string )
                failed_job_text += "Last " + options.num_lines +" line of " + outfile + "\n"
                failed_job_text += "------------------\n"
                count = 0
                for line in reversed(open(outfile).readlines()):
                    failed_job_text +=  line.rstrip() + "\n"
                    count += 1
                    if count == int(options.num_lines):
                        break
                failed_job_text += "------------------\n"
                failed_job_text += "\nContents of " + errfile + "\n"
                failed_job_text += "------------------\n"
                failed_job_text +=  open(errfile).read() + "\n"
                failed_job_text += "------------------\n"
                failed_job_text += "**********************************************************\n"
                a_job_failed = True

        print "-" * len(header)
        if num_passed == num_jobs:
            print "Congratulations! All jobs pass!"

        if a_job_failed:
            failed_job_filename = "failed_job_log_{0}".format(os.path.basename(logfile))
            failed_job_filename = os.path.join(os.path.dirname(logfile), failed_job_filename)
            failed_job_file = open( failed_job_filename, "w" )
            failed_job_file.write( failed_jobs_summary + "\n" )
            failed_job_file.write( failed_job_text )
            failed_job_file.close()
            print "failed job log written to {0}".format(failed_job_filename)

