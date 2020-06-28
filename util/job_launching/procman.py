#!/usr/bin/env python

# 2020 (c) Tim Rogers, Purdue University
# A simple process manager that allow you to queue up tasks and launch them
# once the previous tasks are done. This is useful if you want to launch
# a few hundred tests, but only have 40 cores, you can queue them all up
# using the ProcMan, then let it continuously run 40 at a time.
# This is essentially a poor-man's torque, slurm, condor, etc for a single node.

from optparse import OptionParser
import pickle
import subprocess
from subprocess import Popen, PIPE
import common
import os, signal
import stat
import time

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

class Job:
    def __init__(self, outf, errf, workingDir, command):
        self.outf = outf
        self.errf = errf
        self.workingDir = workingDir
        self.command = command
        self.procId = None
        self.POpenObj = None
        self.status = "NOT_STARTED"

    def string(self):
        return "procId={0},outf={1},errf={2},workingDir={3},command={4},status={5}"\
            .format(self.procId,
                    self.outf,
                    self.errf,
                    self.workingDir,
                    self.command,
                    self.status)

    def __str___(self):
        return self.string()

    def __repr__(self):
        return self.string()


class ProcMan:
    def __init__(self, jobLimit):
        self.initialize(jobLimit)

    def initialize(self, jobLimit):
        self.queuedJobs = []
        self.activeJobs = []
        self.completeJobs = []
        self.spawned = False
        self.jobLimit = jobLimit

    def queueJob(self, job):
        self.queuedJobs.append(job)

    def spawnProcMan(self, outFile, sleepTime):
        if self.spawned:
            exit("Error - can only spawn the procman once")
        else:
            pickle.dump(self,open("procman.pickle", "w+"))
            p = Popen([__file__,"-f", "procman.pickle", "-t", str(sleepTime)],
                stdout=open(outFile,"w+"),
                cwd=this_directory
            )
            print "ProcMan spawned [pid={0}]".format(p.pid)
        self.spawned = True

    def killJobs(self):
        self.queueJob.clear()
        for activeJob in activeJobs:
            # kill the jobs
            pass

    def tick(self):
        idx = 0
        # test jobs for completion
        for activeJob in self.activeJobs:
            jobActive = True
            # for an active session, need to poll or else the thing
            # never dies. If ProcMan is launched with just a file (and
            # did not launch the processes itself, we can just use the kill
            # with CONT signal to see if its still alive.
            if activeJob.POpenObj != None and activeJob.POpenObj.poll() != None:
                jobActive = False
            else:
                try:
                    os.kill(activeJob.procId,signal.SIGCONT)
                except OSError:
                    jobActive = False

            if not jobActive:
                activeJob.status = "COMPLETE"
                self.completeJobs.append(activeJob)
                del self.activeJobs[idx]
            idx += 1

        # launch new jobs when old ones complete
        while len(self.activeJobs) < self.jobLimit and len(self.queuedJobs) > 0:
            newJob = self.queuedJobs.pop(0)
            newJob.POpenObj = Popen(newJob.command,
                stdout=open(newJob.outf,"w+"),
                stderr=open(newJob.errf,"w+"),
                cwd=newJob.workingDir)
            newJob.procId = newJob.POpenObj.pid
            newJob.status = "RUNNING"
            self.activeJobs.append(newJob)

    def updateDisk(self):
        pass

    def getState(self):
        string = "queuedJobs={0}, activeJobs={1}, completeJobs={2}\n"\
            .format(len(self.queuedJobs), len(self.activeJobs), len(self.completeJobs))
        string += "\nqueuedJobs:\n"
        for job in self.queuedJobs:
            string += "\t{0}\n".format(job)

        string += "\nactiveJobs:\n"
        for job in self.activeJobs:
            string += "\t{0}\n".format(job)

        string += "\ncompleteJobs:\n"
        for job in self.completeJobs:
            string += "\t{0}\n".format(job)
        return string

    def complete(self):
        return len(self.queuedJobs) == 0 and len(self.activeJobs) == 0

def selfTest():
    testPath = os.path.join(this_directory, "test")
    if not os.path.isdir(testPath):
        os.makedirs(testPath)
    testOutFile = os.path.join(testPath, "testout.txt")

    jobScript = os.path.join(testPath, "testScript.sh")
    st = os.stat(jobScript)
    os.chmod(jobScript, st.st_mode | stat.S_IEXEC)
    open(jobScript,"w+").write("#!/bin/bash\nsleep 10s")

    print "Starting synchronous selfTest"
    procMan = ProcMan(4)
    for i in range(5):
        procMan.queueJob(
            Job(
                outf=os.path.join(testPath, "out.{0}.txt".format(i)),
                errf=os.path.join(testPath, "err.{0}.txt".format(i)),
                workingDir=testPath,
                command=jobScript
                )
        )
    print procMan.getState()

    while not procMan.complete():
        procMan.tick()
        print procMan.getState()
        time.sleep(3)
    print "Passed synchronous selfTest"

    print "Starting asynchronous selfTest"
    procMan = ProcMan(4)
    for i in range(5):
        procMan.queueJob(
            Job(
                outf=os.path.join(testPath, "out.{0}.txt".format(i)),
                errf=os.path.join(testPath, "err.{0}.txt".format(i)),
                workingDir=testPath,
                command=jobScript
                )
        )
    procMan.spawnProcMan(testOutFile, 3)
    # check the output file for complete
    done = False
    while not done:
        lines = open(testOutFile, "r").readlines()
        if len(lines) > 0:
            lastLine = lines[-1]
            if "All Jobs Complete" == lastLine.strip():
                done = True
        time.sleep(3)
    print "Passed asynchronous selfTest"


def main():
    parser = OptionParser()
    parser.add_option("-s", "--selfTest", dest="selfTest",
                  help="launched the selftester.", action="store_true")
    parser.add_option("-f", "--file", dest="file",
                  help="File with the processes to manage.", default="")
    parser.add_option("-t", "--sleepTime", dest="sleepTime",
                  help="Tune how often. ProcMan looks for completed jobs",
                  type=int, default=30)
    (options, args) = parser.parse_args()

    if options.selfTest:
        selfTest()
    else:
        options.file = common.file_option_test( options.file, "", this_directory )
        if options.file == "":
            exit("Please specify the file containing the processes to manage with -f.")
        procMan = pickle.load(open(options.file))
        while not procMan.complete():
            procMan.tick()
            print procMan.getState()
            time.sleep(options.sleepTime)
        print "All Jobs Complete"

if __name__ == '__main__':
    main()
