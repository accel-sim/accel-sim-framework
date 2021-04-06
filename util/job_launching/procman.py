#!/usr/bin/env python3

# 2020 (c) Tim Rogers, Purdue University
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution. Neither the name of
# Purdue University nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
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

# This file implements ProcMan:
# A simple process manager that allows you to queue up tasks and launch them
# once the previous tasks are done. This is useful if you want to launch
# a few hundred tests, but only have 40 cores, you can queue them all up
# using the ProcMan, then let it continuously run 40 at a time.
# This is essentially a poor-man's torque, slurm, condor, etc for a single node.

# As the goal of proc-man is a dependency-free single-node version of slurm/torque
# it is used in a similar way. Replace:
#   sbatch <file decribing task>
#   qsub <file describing task>
# with
#   ./procmany.py <file describing task>
# After all the processes you want are submitted, start the procman using:
#   ./procman.py -S
# This will create a new process that launchs and manages all the tasks you previously queued
# To avoid inter-process synchronization issues, procman operates using a
# producer/consumer model and once a procman has been started, new work cannot
# be added to that procman. However, more than one procman can run at once. That is,
# you can call:
# ./procman.py mybashA.sh
# ...
# ./procman.py mybashZ.sh
# ./procman.py -S
# ./procman.py mybash1.sh
# ...
# ./procman.py mybash26.sh
# ./procman.py -S
# And it will work, launching 2 procmans that loosely co-ordinate resource usage.
# By default procman will attempt to launch as many jobs as there are cores on the machine
# this can be changes with the "-c <numCores>" option.
#
#   Some other useful commands:
#   ./procman.py -s # launches a self-test to confirm that procman is working (takes 1-2 mins)
#   ./procman.py -p # prints the state of all procmans and their jobs
#   ./procman.py -k # kill all the jobs procman is running
#   ./procman.py --help # prints all the options available
#
# NOTE: procman only works when jobs are submitted from one process. i.e.
#   the user cannot spawn 2 processes and have each process concurrently
#   attempt to enque work. Supporting such a system adds more complication
#   and is not a common case. Also procman is designed for managing one user's
#   processes and knows nothing about what other users are doing on the machine.



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
import glob
import copy

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
procManStateFolder = os.path.join(this_directory, "procman")
procManStateFile = os.path.join(procManStateFolder, "procman.{0}.pickle".format(socket.gethostname().strip()))

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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['POpenObj']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.POpenObj = None

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
    def __init__(self, jobLimit, pickleFile):
        self.initialize(jobLimit, pickleFile)

    def __init__(self, jobLimit):
        self.initialize(jobLimit, procManStateFile)

    def saveState(self):
        pickle.dump(self, open(self.pickleFile, "wb+"))

    def clear(self):
        if not self.mutable or len(self.activeJobs) > 0 or len(self.completeJobs) > 0:
            sys.exit("ProcMans that have been started should not be cleared")
        del self.queuedJobs [:]

    def initialize(self, jobLimit, pickleFile):
        self.queuedJobs = []
        self.activeJobs = {}
        self.completeJobs = {}
        self.jobLimit = int(jobLimit)
        self.nextJobId = 1
        self.tickingProcess = None
        self.mutable = True
        self.pickleFile = pickleFile

    def queueJob(self, job):
        if not self.mutable:
            sys.exit("This ProcMan has already been started. No new jobs can be queued.")
        job.id = self.nextJobId
        self.queuedJobs.append(job)
        self.nextJobId += 1
        return job.id

    def spawnProcMan(self, sleepTime):
        if not self.mutable:
            sys.exit("This ProcMan has already been started. No new spawning can occur.")
        shutil.copy(self.pickleFile, self.pickleFile + ".tmp")
        p = Popen([__file__,"-f", self.pickleFile + ".tmp", "-t", str(sleepTime)],
            cwd=this_directory
        )
        print("ProcMan spawned [pid={0}]".format(p.pid))

    def killJobs(self):
        print("Killing {0} jobs".format(len(self.activeJobs)))
        for jid, activeJob in self.activeJobs.items():
            try:
                p = psutil.Process(activeJob.procId)
            except (psutil.NoSuchProcess,psutil.AccessDenied) as e:
               print(e)
               continue
            for child in p.children(recursive=True):
                os.kill(child.pid,9)
        os.kill(activeJob.procId,9)

    def tick(self):
        if self.tickingProcess == None:
            self.tickingProcess = os.getpid()
            self.pickleFile = self.pickleFile + ".{0}".format(self.tickingProcess)
        elif self.tickingProcess != os.getpid():
            sys.exit("To support concurrent ProcMans in different processes, each procman can only be ticked by one process")

        self.mutable = False
        # test jobs for completion
        jobsMoved = set()
        for jid, activeJob in self.activeJobs.items():
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
                    print(e)
            else:
                activeJob.status = "COMPLETE_NO_OTHER_INFO"
                self.completeJobs[activeJob.id] = activeJob
                jobsMoved.add(activeJob.id)

        for jobId in jobsMoved:
            del self.activeJobs[jobId]

        othersCores = self.getCPUCountFromOtherProcMans()

        # launch new jobs when old ones complete
        while (len(self.activeJobs) + othersCores) < self.jobLimit and len(self.queuedJobs) > 0:
            newJob = self.queuedJobs.pop(0)
            newJob.POpenObj = Popen(newJob.command,
                stdout=open(newJob.outF,"w+"),
                stderr=open(newJob.errF,"w+"),
                cwd=newJob.workingDir)
            newJob.procId = newJob.POpenObj.pid
            newJob.hostname = socket.gethostname().strip()
            newJob.status = "RUNNING"
            self.activeJobs[newJob.id] = newJob

    def getCPUCountFromOtherProcMans(self):
        othersCores = 0
        for pickleFile in glob.glob(os.path.join(os.path.dirname(self.pickleFile),"*pickle*")):
            if pickleFile != self.pickleFile:
                otherProcMan = pickle.load(open(pickleFile, 'rb'))
                othersCores += len(otherProcMan.activeJobs)
        return othersCores

    def getState(self):
        string = "queuedJobs={0}, activeJobs={1}, completeJobs={2}\n"\
            .format(len(self.queuedJobs), len(self.activeJobs), len(self.completeJobs))
        string += "\nqueuedJobs:\n"
        for job in self.queuedJobs:
            string += "\t{0}\n".format(job)

        string += "\nactiveJobs:\n"
        for jid,job in self.activeJobs.items():
            string += "\t{0}\n".format(job)

        string += "\ncompleteJobs:\n"
        for jid,job in self.completeJobs.items():
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

    print("Starting synchronous selfTest")
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
    print(procMan.getState())

    while not procMan.complete():
        procMan.tick()
        print(procMan.getState())
        time.sleep(3)
    print("Passed synchronous selfTest")

    print("Starting asynchronous selfTest")
    for i in range(int(psutil.cpu_count()*1.2)):
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
        out = out.decode("utf-8")
        if err != None:
            err = err.decode("utf-8")
            sys.exit(err)
        print("Queued Job {0}".format(out))

    print("Starting Jobs")
    subprocess.Popen([os.path.join(this_directory, "procman.py"),\
                                        "-S", "-t", "5"], stdout=PIPE)
    out = ""
    while out != "Nothing Active":
        time.sleep(1)
        out, err = subprocess.Popen([os.path.join(this_directory, "procman.py"),\
                                        "-p"], stdout=PIPE).communicate()
        out = out.strip().decode("utf-8")
        if err != None:
            err = err.decode("utf-8")
            sys.exit(err)
        print(out)
    print("Asynchronous test passed")


    print("Starting multi ProcMan test")
    JOBS_PER_PROCMAN = int(int(psutil.cpu_count()*1.2) / 4)
    for j in range(4):
        for i in range(JOBS_PER_PROCMAN):
            jobNum = j*JOBS_PER_PROCMAN + i
            jobScript = os.path.join(testPath, "testSlurm.{0}.sh".format(jobNum))
            open(jobScript,"w+").write("#!/bin/bash\n"\
                                       "#SBATCH -J test.{0}\n".format(jobNum) +\
                                       "#SBATCH --output={0}\n".format(os.path.join(testPath, "out.{0}.txt".format(jobNum))) +\
                                       "#SBATCH --error={0}\n".format(os.path.join(testPath, "out.{0}.txt".format(jobNum))) +\
                                       "sleep 20s")
            st = os.stat(jobScript)
            os.chmod(jobScript, st.st_mode | stat.S_IEXEC)
            out, err = subprocess.Popen([os.path.join(this_directory, "procman.py"),\
                                            jobScript], stdout=PIPE).communicate()
            out = out.decode("utf-8")
            if err != None:
                err = err.decode("utf-8")
                sys.exit(err)
            print("ProcMan {0}: Queued Job {0}".format(j, out))

        print("ProcMan {0}: Starting Jobs".format(j))
        subprocess.Popen([os.path.join(this_directory, "procman.py"),\
                                             "-S", "-t", "5"], stdout=PIPE)

    out = ""
    while out != "Nothing Active":
        time.sleep(1)
        out, err = subprocess.Popen([os.path.join(this_directory, "procman.py"),\
                                        "-p"], stdout=PIPE).communicate()
        out = out.strip().decode("utf-8")
        if err != None:
            err = err.decode("utf-8")
            sys.exit(err)
        print(out)
    print("Multi-ProcMan test passed")
    shutil.rmtree(testPath)


def main():
    parser = OptionParser()
    parser.add_option("-s", "--selfTest", dest="selfTest",
                  help="launched the selftester.", action="store_true")
    parser.add_option("-f", "--file", dest="file",
                  help="File with the processes to manage.", default=procManStateFile)
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
    parser.add_option("-j", "--procManForJob", dest="procManForJob",default=None, type=int,
                  help="Return the path of the pickle file for the ProcMan managing this job." )
    (options, args) = parser.parse_args()

    if options.selfTest:
        selfTest()
    elif options.kill:
        procmanfiles = glob.glob(options.file + ".*")
        for f in procmanfiles:
            print("Killing active jobs in Procman: {0}".format(os.path.basename(f)))
            procMan = pickle.load(open(f, 'rb'))
            procMan.killJobs()
    elif options.printState:
        numProcMans = 0
        numQueued = 0
        numActive = 0
        numComplete = 0
        procmanfiles = glob.glob(options.file + ".*")
        if len(procmanfiles) == 0:
            print("Nothing Active")
        else:
            for f in procmanfiles:
                numProcMans += 1
                procMan = pickle.load(open(f, 'rb'))
                numQueued += len(procMan.queuedJobs)
                numActive += len(procMan.activeJobs)
                numComplete += len(procMan.completeJobs)
                print("Procman: {0}".format(os.path.basename(f)))
                print(procMan.getState())
            print("Total Procmans={0}, Total Queued={1}, Total Running={2}, Total Complete={3}"\
                .format(numProcMans, numQueued, numActive, numComplete))
    elif options.start:
        if not os.path.exists(options.file):
             sys.exit("Nothing to start {0} does not exist".format(options.file))
        procMan = pickle.load(open(options.file, 'rb'))
        if procMan.jobLimit != int(options.cores):
            procMan.jobLimit = int(options.cores)
            procMan.saveState()
        procMan.spawnProcMan(options.sleepTime)
        procMan.clear()
        procMan.saveState()
    elif options.procManForJob != None:
        procmanfiles = glob.glob(options.file + ".*")
        for f in procmanfiles:
            procMan = pickle.load(open(f, 'rb'))
            j = procMan.getJob(options.procManForJob)
            if j != None:
                print(procMan.pickleFile)
                break
    elif len(args) == 1:
        # To make this work the same as torque and slurm - if you just give it one argument,
        # we assume it's a pointer to a job file you want to submit.
        if os.path.exists(options.file):
            procMan = pickle.load(open(options.file, 'rb'))
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
        procMan.saveState()
        print(job.id)
    else:
        options.file = common.file_option_test( options.file, "", this_directory )
        if options.file == "":
            sys.exit("Please specify the file containing the processes to manage with -f.")
        procMan = pickle.load(open(options.file, 'rb'))
        if procMan.tickingProcess!= None:
            sys.exit("This procman is already running {0}".format(os.path.basename(options.file)))
        procMan.pickleFile = options.file
        os.remove(options.file)
        if len(procMan.queuedJobs) > 0:
            while not procMan.complete():
                procMan.tick()
                procMan.saveState()
                time.sleep(options.sleepTime)
            os.remove(procMan.pickleFile)

if __name__ == '__main__':
    if not os.path.exists(procManStateFolder):
        os.makedirs(procManStateFolder)
    main()
