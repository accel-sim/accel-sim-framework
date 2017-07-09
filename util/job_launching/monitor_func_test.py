#!/usr/bin/env python

from optparse import OptionParser
import re
import os
import subprocess
import sys
import common
import time

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

(options, args) = parser.parse_args()
options.logfile = options.logfile.strip()
options.sim_name = options.sim_name.strip()


jobstatus_out_filename = "/tmp/job_status_out.txt"

while True:
    print "Beat"
    jobstatus_out_file = open(jobstatus_out_filename, 'w+')
    print "File opened"
    if subprocess.call([os.path.join(this_directory, "job_status.py") ,"-l", options.logfile, "-N", options.sim_name],
        stdout=jobstatus_out_file, stderr=jobstatus_out_file) < 0:
            exit("Error Launching job_status.py")
    else:
        jobstatus_out_file.seek(0)
        jobStatusCol = None
        num_passed = 0
        num_not_done = 0
        num_else = 0
        for line in jobstatus_out_file.readlines():
            print line
            if jobStatusCol == None:
                name_line_match = re.match("(.*)JobStatus.*", line)
                if name_line_match != None:
                    jobStatusCol = name_line_match.group(1).count('\t')
                    continue
            else:
                tokens = line.split('\t')
                if len(tokens) > jobStatusCol:
                    status = tokens[jobStatusCol].strip()
                    if status == "FUNC_TEST_PASSED":
                        num_passed += 1
                    elif status == "RUNNING" or status == "WAITING_TO_RUN":
                        num_not_done += 1
                    else:
                        num_else += 1
                elif re.match("failed job log.*", line):
                    failed_job_file = line

        jobstatus_out_file.close()
        os.remove(jobstatus_out_filename)
    
    total = num_passed + num_not_done + num_else
    print "Passed:{0}/{1}, Not passed:{2}/{1}, Not done:{3}/{1}"\
        .format(num_passed, total, num_else, num_not_done)
    if num_else > 0:
            print failed_job_file

    if num_not_done == 0:
        print "All {0} Tests Done.".format(total)
        if num_else == 0:
            print "Congratulations! All Tests Pass!"
            exit(0)
        else:
            print "Something did not pass."
            exit(1)
    else:
        print "Sleeping for 30s"
        time.sleep(30)
