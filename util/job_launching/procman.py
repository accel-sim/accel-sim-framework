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
import psutil
import shutil
import datetime
import re
import socket
import sys

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
procManStateFile = os.path.join(this_directory,"procman.{0}.pickle".format(socket.gethostname().strip()))

class Job:
    def __init__(self, outF, errF, workingDir, command):
        self.outF = outF
        self.errF = errF
        self.workingDir = workingDir
        self.command = command
        self.procId = None
        self.POpenObj = None
        self.maxVmSize = 0
        self.runningTime = 0
        self.status = "WAITING_TO_RUN"
        self.name = None
        self.id = None
        self.hostname = "UNKNOWN"

    def string(self):
        return "status={0}: [name={8},procId={1},maxVmSize={2},runningTime={3},outF={4}," \
            "errF={5},workingDir={6},command={7}]"\
            .format(self.status,
                    self.procId,
                    self.maxVmSize,
                    self.runningTime,
                    self.outF,
                    self.errF,
                    self.workingDir,
                    self.command,
                    self.name)

    def __str__(self):
        return self.string()

    def __repr__(self):
        return self.string()


class ProcMan:
    def __init__(self, jobLimit):
        self.initialize(jobLimit)

    def initialize(self, jobLimit):
        self.queuedJobs = []
        self.activeJobs = {}
        self.completeJobs = {}
        self.jobLimit = jobLimit
        self.nextJobId = 1
        self.tickingProcess = None
        self.mutable = True

    def queueJob(self, job):
        if not self.mutable:
            sys.exit("This ProcMan has already been started. No new jobs can be queued.")
        job.id = self.nextJobId
        self.queuedJobs.append(job)
        self.nextJobId += 1
        return job.id

    def spawnProcMan(self, pickleFile, sleepTime):
        if not self.mutable:
            sys.exit("This ProcMan has already been started. No new spawning can occur.")
        p = Popen([__file__,"-f", pickleFile, "-t", str(sleepTime)],
            cwd=this_directory
        )
        print "ProcMan spawned [pid={0}]".format(p.pid)

    def killJobs(self):
        print "Killing {0} jobs".format(len(self.activeJobs))
        for jid, activeJob in self.activeJobs.iteritems():
            try:
                p = psutil.Process(activeJob.procId)
            except (psutil.NoSuchProcess,psutil.AccessDenied) as e:
               print e
               continue
            for child in p.children(recursive=True):
                os.kill(child.pid,9)
        os.kill(activeJob.procId,9)

    def tick(self):
        if self.tickingProcess != None and self.tickingProcess != os.getpid():
            sys.exit("To support concurrent ProcMans in different processes, each procman can only be ticked by one process")
        self.tickingProcess = os.getpid()
        self.mutable = False
        # test jobs for completion
        jobsMoved = set()
        for jid, activeJob in self.activeJobs.iteritems():
            jobActive = True
            # for an active session, need to poll or else the thing
            # never dies. If ProcMan is launched with just a file (and
            # did not launch the processes itself, we can just use the kill
            # with CONT signal to see if its still alive.
            if activeJob.POpenObj != None:
                activeJob.POpenObj.poll()
            try:
                os.kill(activeJob.procId,signal.SIGCONT)
            except OSError:
                jobActive = False

            if jobActive:
                try:
                    p = psutil.Process(activeJob.procId)
                    mem = p.memory_info().vms
                    for child in p.children(recursive=True):
                        mem += child.memory_info().vms
                    activeJob.maxVmSize = max(mem, activeJob.maxVmSize)
                    activeJob.runningTime = \
                        datetime.datetime.now() \
                            - datetime.datetime.fromtimestamp(p.create_time())
                    activeJob.runningTime = str(activeJob.runningTime).split('.')[0]
                except (psutil.NoSuchProcess,psutil.AccessDenied) as e:
                    print e
            else:
                activeJob.status = "COMPLETE_NO_OTHER_INFO"
                self.completeJobs[activeJob.id] = activeJob
                jobsMoved.add(activeJob.id)

        for jobId in jobsMoved:
            del self.activeJobs[jobId]

        # launch new jobs when old ones complete
        while len(self.activeJobs) < self.jobLimit and len(self.queuedJobs) > 0:
            newJob = self.queuedJobs.pop(0)
            newJob.POpenObj = Popen(newJob.command,
                stdout=open(newJob.outF,"w+"),
                stderr=open(newJob.errF,"w+"),
                cwd=newJob.workingDir)
            newJob.procId = newJob.POpenObj.pid
            newJob.hostname = socket.gethostname().strip()
            newJob.status = "RUNNING"
            self.activeJobs[newJob.id] = newJob

    def getState(self):
        string = "queuedJobs={0}, activeJobs={1}, completeJobs={2}\n"\
            .format(len(self.queuedJobs), len(self.activeJobs), len(self.completeJobs))
        string += "\nqueuedJobs:\n"
        for job in self.queuedJobs:
            string += "\t{0}\n".format(job)

        string += "\nactiveJobs:\n"
        for jid,job in self.activeJobs.iteritems():
            string += "\t{0}\n".format(job)

        string += "\ncompleteJobs:\n"
        for jid,job in self.completeJobs.iteritems():
            string += "\t{0}\n".format(job)
        return string

    def getJob(self, jobId):
        if jobId in self.activeJobs:
            return self.activeJobs[jobId]
        elif jobId in self.completeJobs:
            return self.completeJobs[jobId]
        else:
            for job in self.queuedJobs:
                if jobId == job.id:
                    return job
        return None

    def complete(self):
        return len(self.queuedJobs) == 0 and len(self.activeJobs) == 0

def selfTest():
    testPath = os.path.join(this_directory, "test")
    if not os.path.isdir(testPath):
        os.makedirs(testPath)

    jobScript = os.path.join(testPath, "testScript.sh")
    open(jobScript,"w+").write("#!/bin/bash\nsleep 20s")
    st = os.stat(jobScript)
    os.chmod(jobScript, st.st_mode | stat.S_IEXEC)

    print "Starting synchronous selfTest"
    procMan = ProcMan(4)
    for i in range(5):
        procMan.queueJob(
            Job(
                outF=os.path.join(testPath, "out.{0}.txt".format(i)),
                errF=os.path.join(testPath, "err.{0}.txt".format(i)),
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
    for i in range(5):
        jobScript = os.path.join(testPath, "testSlurm.{0}.sh".format(i))
        open(jobScript,"w+").write("#!/bin/bash\n"\
                                   "#SBATCH -J test.{0}\n".format(i) +\
                                   "#SBATCH --output={0}\n".format(os.path.join(testPath, "out.{0}.txt".format(i))) +\
                                   "#SBATCH --error={0}\n".format(os.path.join(testPath, "out.{0}.txt".format(i))) +\
                                   "sleep 20s")
        st = os.stat(jobScript)
        os.chmod(jobScript, st.st_mode | stat.S_IEXEC)
        out, err = subprocess.Popen([os.path.join(this_directory, "procman.py"),\
                                        jobScript], stdout=PIPE).communicate()
        print "Launched Job {0}".format(out)

    out, errt = subprocess.Popen([os.path.join(this_directory, "procman.py"),\
                                        "-S"], stdout=PIPE).communicate()
    print out
    err += errt

    procmanfiles = glob.glob(os.path.join(testPath, "procMan.*.pickle.*"))
    out = ""
    while out != "Nothing Active":
        time.sleep(3)
        out, errt = subprocess.Popen([os.path.join(this_directory, "procman.py"),\
                                        "-p"], stdout=PIPE).communicate()
        print out
        err += errt
    if len(err) == 0:
        print "Passed asynchronous selfTest"
    else:
        print "Asynchrnous seftTest failed. Errors:\n\n{0}".format(err)
    shutil.rmtree(testPath)


def main():
    parser = OptionParser()
    parser.add_option("-s", "--selfTest", dest="selfTest",
                  help="launched the selftester.", action="store_true")
    parser.add_option("-f", "--file", dest="file",
                  help="File with the processes to manage.", default="")
    parser.add_option("-t", "--sleepTime", dest="sleepTime",
                  help="Tune how often. ProcMan looks for completed jobs",
                  type=int, default=30)
    parser.add_option("-c", "--cores", dest="cores",
                  help="how many cores to use",
                  type=int, default=psutil.cpu_count())
    parser.add_option("-S", "--start", dest="start",action="store_true",
                  help="Just spawn the manager")
    parser.add_option("-p", "--printState", dest="printState",action="store_true",
                  help="Print the state of the manager")
    parser.add_option("-k", "--kill", dest="kill",action="store_true",
                  help="Kill all managed processes")
#    parser.add_option("-P", "--pickleFileQuerry", dest="pickleFileQuerry",action="store_true",
#                  help="Return the path of the pickle file storing the persistent state. " \
#                       "This is useful if you want to create a copy of the ProcMan to look around " \
#                       "in another process" )
    (options, args) = parser.parse_args()

    if options.selfTest:
        selfTest()
    elif options.kill:
        procmanfiles = glob.glob(procManStateFile + ".*")
        for f in procmanfiles:
            print "Killing active jobs in Procman: {0}".format(os.path.filename(f))
            procMan = pickle.load(open(f))
            procMan.killJobs()
    elif options.printState:
        procmanfiles = glob.glob(procManStateFile + ".*")
        if len(procmanfiles) == 0:
            print "Nothing Active"
        for f in procmanfiles:
            procMan = pickle.load(open(f))
            print "Procman: {0}".format(os.path.filename(f))
            print procMan.getState()
    elif options.start:
        if not os.path.exists(procManStateFile):
             sys.exit("Nothing to start {0} does not exist".format(procManStateFile))
        procMan = pickle.load(open(procManStateFile))
        procMan.spawnProcMan(procManStateFile, 10)
        os.rm(procManStateFile)
    elif len(args) == 1:
        # To make this work the same as torque and slurm - if you just give it one argument,
        # we assume it's a pointer to a job file you want to submit.
        if os.path.exists(procManStateFile):
            procMan = pickle.load(open(procManStateFile))
            if not procMan.mutable:
                sys.exit("Error - this procman has already started")
        else:
            procMan = ProcMan(options.cores)
        exec_file = args[0]
        st = os.stat(exec_file)
        os.chmod(exec_file, st.st_mode | stat.S_IEXEC)

        # slurmToJob
        job = Job("","",os.getcwd(),exec_file)
        job.id = procMan.queueJob(job)
        contents = ""
        for line in open(exec_file).readlines():
            if line.startswith("#SBATCH"):
                nameMatch = re.match(r"#SBATCH -J (.*)", line.strip())
                if nameMatch:
                    job.name = nameMatch.group(1)
                outFMatch = re.match(r"#SBATCH --output=(.*)", line.strip())
                if outFMatch:
                    job.outF = outFMatch.group(1)
                errFMatch = re.match(r"#SBATCH --error=(.*)", line.strip())
                if errFMatch:
                    job.errF = errFMatch.group(1)
            line = re.sub(r"\$SLURM_JOB_ID", str(job.id), line)
            contents += line
        with open(exec_file, "w+") as f:
            f.write(contents)

        job.outF = re.sub("\%j", str(job.id), job.outF)
        job.errF = re.sub("\%j", str(job.id), job.errF)
        pickle.dump(procMan, open(procManStateFile,"w+"))
        print job.id
    else:
        options.file = common.file_option_test( options.file, "", this_directory )
        if options.file == "":
            sys.exit("Please specify the file containing the processes to manage with -f.")
        procMan = pickle.load(open(options.file))
        if procMan.tickingProcess!= None:
            sys.exit("This procman is already running {0}".format(os.path.filename(options.file)))
        while not procMan.complete():
            procMan.tick()
            pickle.dump(procMan, open(options.file + ".{0}".format(procMan.tickingProcess), "w+"))
            time.sleep(options.sleepTime)
        shutil.rm(options.file + ".{0}".format(procMan.tickingProcess))

if __name__ == '__main__':
    main()
