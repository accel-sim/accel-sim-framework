#!/usr/bin/env python3

from optparse import OptionParser
import re
import os
import subprocess
import sys
import common
import time

def getColId(colHeader, headerLine):
    name_line_match = re.match(colHeader, headerLine)
    if name_line_match != None:
        return name_line_match.group(1).count('\t')
    return None

def kill_all_running_jobs(jobstatus_out_filename):
    jobstatus_out_file = open(jobstatus_out_filename, 'r')
    jobStatusCol = None
    torqueJobCol = None
    appCol = None
    appArgsCol = None
    nodeCol = None
    for line in jobstatus_out_file.readlines():
        if jobStatusCol == None:
            jobStatusCol = getColId("(.*)JobStatus.*", line)
            torqueJobCol = getColId("(.*)TorqueJob.*", line)
            appCol = getColId("(.*)App\s+.*", line)
            appArgsCol = getColId("(.*)AppArgs.*", line)
            nodeCol = getColId("(.*)Node.*", line)
        else:
            tokens = line.split('\t')
            if len(tokens) > jobStatusCol:
                status = tokens[jobStatusCol].strip()
                if status == "RUNNING" or status == "WAITING_TO_RUN":
                    torqueID = tokens[torqueJobCol].strip()
                    app = tokens[appCol].strip()
                    appArgs = tokens[appArgsCol].strip()
                    nodeName = tokens[nodeCol].strip()
                    print("Calling qdel {0}: {1}/{2} ({3})".format(torqueID, app, appArgs, nodeName))
                    if subprocess.call( ["qdel" , torqueID] ) != 0:
                        print("WARNING error calling qdel")
    print("Sleeping 30s to wait for the file system to calm down")
    time.sleep(30)
    jobstatus_out_file.close()

def print_statsfile(options, this_directory):
    get_stats_out_file = open(options.statsfile, 'w+')
    print("Calling get_stats.py")
    if subprocess.call([os.path.join(this_directory, "get_stats.py") ,"-R" ,"-l", options.logfile, "-N", options.sim_name, "-A"],
        stdout=get_stats_out_file, stderr=get_stats_out_file) != 0:
        print("Error Launching get_stats.py")
    get_stats_out_file.seek(0)
    print(get_stats_out_file.read())
    get_stats_out_file.close()

def handle_exit(jobstatus_out_filename):
    os.remove(jobstatus_out_filename)
    if num_error == 0 or options.ignore_failures:
        if options.verbose and options.statsfile:
            print_statsfile(options, this_directory)
        exit(0)
    else:
        exit(1)

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
parser.add_option("-N", "--sim_name", dest="sim_name",
                  help="If you are launchign run_simulations.py with the \"-N\" option" +\
                       " then you can run ./job_status.py with \"-N\" and it will" + \
                       " give you the status of the latest run with that name."+ \
                       " if you want older runs from this name, then just point it directly at the"+\
                       " logfile with \"-l\"", default="")
parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                  help="Constantly print stuff")
parser.add_option("-s", "--statsfile", dest="statsfile", default="",
                  help="In verbose mode specify where the stats go")
parser.add_option("-S", "--sleep_time", dest="sleep_time", default="30",
                  help="Time to sleep in (s) - default is 30.")
parser.add_option("-I", "--ignore_failures", dest="ignore_failures", action="store_true",
                  help="If some of the runs have errors - do not return an error code.")
parser.add_option("-T", "--timeout", dest="timeout", default="99999",
                  help="Maximum number of hours to run the monitor for. If time is exceeded, exit with error code.")
parser.add_option("-K", "--killwhentimedout", dest="killwhentimedout", action="store_true",
                  help="Kill all the jobs still running if the monitor times out.")
parser.add_option("-j", "--job_manager", dest="job_manager",
                  help="Pick between slurm, torque and out local procman. If not specified, "\
                       "we will select it automatically.", default=None)

(options, args) = parser.parse_args()
options.logfile = options.logfile.strip()
options.sim_name = options.sim_name.strip()
options.timeout = float(options.timeout) * 60 * 60

if options.job_manager != None:
    job_manager = options.job_manager
elif any([os.path.isfile(os.path.join(p, "squeue")) for p in os.getenv("PATH").split(os.pathsep)]):
    job_manager = "squeue"
elif any([os.path.isfile(os.path.join(p, "qstat")) for p in os.getenv("PATH").split(os.pathsep)]):
    job_manager = "qstat"
else:
    job_manager = "procman"

jobstatus_out_filename = os.path.join(this_directory, "job_status_out-{0}.txt".format(os.getpid()))
failed_job_file = ""

while True:
    jobstatus_out_file = open(jobstatus_out_filename, 'w+')
    if options.verbose:
        print("Calling job_status.py")
    if subprocess.call([os.path.join(this_directory, "job_status.py"),\
            "-l", options.logfile, \
            "-N", options.sim_name, \
            "-j", job_manager],\
            stdout=jobstatus_out_file, stderr=jobstatus_out_file) != 0:
        jobstatus_out_file.seek(0)
        print(jobstatus_out_file.read())
        exit("Error Launching job_status.py")
    else:
        jobstatus_out_file.seek(0)
        jobStatusCol = None
        num_passed = 0
        num_running = 0
        num_waiting = 0
        num_error = 0
        num_no_err = 0
        for line in jobstatus_out_file.readlines():
            if options.verbose:
                print(line.strip())
            if jobStatusCol == None:
                jobStatusCol = getColId("(.*)JobStatus.*", line)
            else:
                tokens = line.split('\t')
                fail_match = re.match("failed job log written to (.*)", line)
                if fail_match != None:
                    failed_job_file = fail_match.group(1)
                elif len(tokens) > jobStatusCol:
                    status = tokens[jobStatusCol].strip()
                    if status == "FUNC_TEST_PASSED":
                        num_passed += 1
                    elif status == "COMPLETE_NO_OTHER_INFO":
                        num_no_err += 1
                    elif status == "RUNNING":
                        num_running += 1
                    elif status == "WAITING_TO_RUN":
                        num_waiting += 1
                    else:
                        num_error += 1

        jobstatus_out_file.close()
    
    total = num_passed + num_running + num_waiting + num_error + num_no_err
    print("Passed:{0}/{1}, No error:{2}/{1}, Failed/Error:{3}/{1}, Running:{4}/{1}, Waiting:{5}/{1}"\
        .format(num_passed, total, num_no_err, num_error, num_running, num_waiting))
    if num_error > 0:
        print("Contents {0}:".format(failed_job_file))
        if options.verbose and os.path.exists(failed_job_file):
            print(open(failed_job_file).read())

    if num_running + num_waiting == 0:
        print("All {0} Tests Done.".format(total))
        if num_error == 0:
            print("Congratulations! All Tests Pass!")
        else:
            print("Something did not pass.")

        handle_exit(jobstatus_out_filename)
    else:
        print("Sleeping for {0}s".format(options.sleep_time))
        time.sleep(int(options.sleep_time))
        options.timeout -= float(options.sleep_time)
        if options.timeout <= 0:
            print("Monitor has timed-out")
            if options.killwhentimedout:
                kill_all_running_jobs(jobstatus_out_filename)
            handle_exit(jobstatus_out_filename)
