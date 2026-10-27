#!/usr/bin/env python3
"""Status of the simulations of a run_simulations.py launch.

    job_status.py [-N <launch name> | -l <logfile> | -l all] [-r <run dir>]
                  [-j squeue|qstat|procman] [-n <lines>]

Prints one row per job and, when a job failed, writes the failed jobs' output
tails to logfiles/failed_job_log_<logfile>. monitor_func_test.py calls
status() once per poll.

A job is done when its scheduler neither runs nor queues it. A done job
fails if Slurm ended it in any state but COMPLETED (sacct: squeue forgets a
finished job within minutes, sacct keeps it), if its output has a failure
token, if it wrote to stderr, or if it left no output. qstat and procman
report no exit state, so there only the output decides. A done job is
judged once and cached in logfiles/node_details.<launch>.
"""

from optparse import OptionParser
import json
import os
import pickle
import re
import subprocess
import sys
import time

import common

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

RUNNING, WAITING = "RUNNING", "WAITING_TO_RUN"
PASSED, NO_ERROR = "FUNC_TEST_PASSED", "COMPLETE_NO_OTHER_INFO"
NO_OUTPUT, ERR_FILE = "NOT_RUNNING_NO_OUTPUT", "COMPLETE_ERR_FILE_HAS_CONTENTS"

# Searched for in the tail of a done job's .o and .e files; the first entry
# marks the application's own functional check as passed, the rest a failure
PASS_TOKEN = re.compile(r"passed|PASSED")
FAIL_TOKENS = {
    "Assertion": "ASSERT",
    "Segmentation fault": "SEGF",
    "Aborted": "ABORTED",
    "PBS": "TORQUE_KILLED",
    "deadlock.*detected": "DEADLOCK",
    "failed to open display": "FAIL_NO_DISPLAY",
    "FAILED|[Ff]ailed": "FUNC_TEST_FAILED",
    "XML Parsing error.*gpuwattch": "NO_GPU_WATTCH_CFG",
    # GPGPU-Sim's own error exits, e.g. "GPGPU-Sim PTX: ERROR ** while loading PTX"
    r"ERROR \*\*": "SIM_ERROR",
}
TAIL_BYTES = 1 << 20

STATS_TO_PULL = {
    "SIM_TIME": r"gpgpu_simulation_time\s*=[^1-9]*(.*)",
    "TOT_INSN": r"gpu_tot_sim_insn\s*=\s*(.*)",
    "TOT_IPC": r"gpu_tot_ipc\s*=\s*(.*)",
    "TOT_CYCLE": r"gpu_tot_sim_cycle\s*=\s*(.*)",
    "SIMRATE_IPS": r"gpgpu_simulation_rate\s*=\s*(.*)\s*\(inst/sec\)",
}

ROW_STRING = (
    "{jobId:<16.16}\t{exec_node:<30.30}\t{app:<20.20}\t{args:<20.20}\t"
    + "{version:20.20}\t{config:10.10}\t{running_time:15}\t{mem_used:6}\t{status:30.30}\t"
    + "{stat:50}\t"
)

# A job missing from both squeue and sacct this many polls in a row (sacct lags
# a job's end by seconds), or whose procman state is unknown, is judged
# without a scheduler state
SACCT_MISSES = 3
# procman_state() for a job whose procman pickle could not be looked up or read
UNKNOWN = "UNKNOWN"
# sacct states of a job that is not over yet
SLURM_ACTIVE = ("", "PENDING", "RUNNING", "REQUEUED", "COMPLETING", "CONFIGURING",
                "SUSPENDED", "RESIZING", "REQUEUE_HOLD", "REQUEUE_FED")


def pick_job_manager(name=None):
    if name:
        return name
    for tool, manager in (("squeue", "squeue"), ("qstat", "qstat")):
        if any(os.path.isfile(os.path.join(p, tool)) for p in os.getenv("PATH").split(os.pathsep)):
            return manager
    return "procman"


def millify(n):
    """'12345678' -> '12 M' (sacct's and procman's sizes keep their own units)."""
    try:
        v = float(n)
    except (TypeError, ValueError):
        return str(n)
    for unit in ("", " K", " M", " G"):
        if abs(v) < 1000:
            return "{:.0f}{}".format(v, unit)
        v /= 1000
    return "{:.0f} T".format(v)


# ---------------------------------------------------------------------------
# Logfiles and jobs
# ---------------------------------------------------------------------------


def find_logfiles(logfile, sim_name):
    logdir = os.path.join(this_directory, "logfiles")
    if logfile == "all":
        return sorted(os.path.join(logdir, f) for f in os.listdir(logdir) if re.match(r"sim_log.*\.latest", f))
    if logfile:
        return [common.file_option_test(logfile, "", this_directory)]
    if not os.path.isdir(logdir):
        exit("No logfile specified and the default logfile directory cannot be found")
    logs = [os.path.join(logdir, f) for f in os.listdir(logdir) if f.startswith("sim_log")]
    if sim_name:
        logs = [l for l in logs if re.match(r".*/sim_log\.{0}\.".format(re.escape(sim_name)), l)]
        if not logs:
            exit('Could not find logfiles for job with the name "{0}"'.format(sim_name))
    if not logs:
        exit("ERROR - No Logfiles in " + logdir)
    return [max(logs, key=os.path.getmtime)]


def parse_logfile(logfile, run_dir):
    """The jobs of a launch log: one "<time> <id> <app> <args> <config> <jobname>"
    line each. --per-kernel jobs log args as "<argfolder>/<kernel-N>" and run in
    <app>/<argfolder>/<config>/per-kernel/<kernel-N>/."""
    jobs, warnings = [], []
    for line in open(logfile):
        fields = line.split()
        if len(fields) != 6:
            warnings.append('logfile line "{0}" is missing fields: its launch likely failed'.format(line.strip()))
            continue
        _, job_id, app, args, config, jobname = fields
        base_args, kernel = args, None
        pk = re.match(r"(.*)/(kernel-[^/]+)$", args)
        if pk:
            base_args, kernel = pk.groups()
        outdir = os.path.join(run_dir, app.replace("/", "_"), base_args, config)
        if kernel:
            outdir = os.path.join(outdir, "per-kernel", kernel)
        if not os.path.isdir(outdir):
            warnings.append("the output dir {0} does not exist".format(outdir))
            continue
        # Must match the output names run_simulations.py gives the job script
        base = re.sub(r".*\.([^\s]*-commit-.*-commit-.*)", r"\1", jobname)
        name = (os.path.basename(app) + "-" + base_args + "." + base + ("." + kernel if kernel else ""))[:200]
        jobs.append({
            "id": job_id, "app": app, "args": args, "config": config,
            "version": re.sub(r".*-commit-([^_]{7})[^_]+_(.*)\.so", r"\1-\2", jobname),
            "outfile": os.path.join(outdir, name + ".o" + job_id),
            "errfile": os.path.join(outdir, name + ".e" + job_id),
        })
    return jobs, warnings


# ---------------------------------------------------------------------------
# Schedulers: live state of the jobs not yet judged
# ---------------------------------------------------------------------------


def slurm_live():
    """{job id: (state, nodes, time)} of every queued or running job and array task."""
    out = subprocess.check_output(["squeue", "-h", "-r", "-o", "%i|%t|%N|%M"]).decode()
    live = {}
    for line in out.splitlines():
        jid, state, nodes, elapsed = line.strip().split("|")
        live[jid] = (RUNNING if state in ("R", "CG", "CF", "S") else WAITING, nodes, elapsed)
    return live


def qstat_state(job_id):
    out = subprocess.run(["qstat", "-f", job_id], capture_output=True, text=True).stdout
    out = re.sub("\n", " ", out)
    m = re.search(r"job_state\s=\s([^\s]*)", out)
    if not m:
        return None, "", ""
    host = re.search(r"exec_host\s=\s([^\s]*)", out)
    walltime = re.search(r"resources_used.walltime\s=\s([^\s]*)", out)
    state = {"R": RUNNING, "E": RUNNING, "Q": WAITING, "H": WAITING, "W": WAITING}.get(m.group(1))
    return state, host.group(1) if host else "", walltime.group(1) if walltime else ""


class _ProcManUnpickler(pickle.Unpickler):
    # procman.py pickles its state as __main__.ProcMan / __main__.Job
    def find_class(self, module, name):
        return super().find_class("procman" if module == "__main__" else module, name)


def procman_state(job_id):
    """(state, node, time, mem); state None = no procman has the job, UNKNOWN = the lookup
    or the pickle read failed, which is not evidence that the job is done."""
    r = subprocess.run([os.path.join(this_directory, "procman.py"), "-j", job_id],
                       capture_output=True, text=True)
    out = r.stdout.strip()
    if r.returncode != 0:
        return UNKNOWN, "", "", ""
    if not out:  # its procman finished, or is still taking over the launcher's jobs
        return None, "", "", ""
    try:
        with open(out, "rb") as f:
            job = _ProcManUnpickler(f).load().getJob(int(job_id))
    except Exception:
        return UNKNOWN, "", "", ""
    if job is None:
        return None, "", "", ""
    state = job.status if job.status in (RUNNING, WAITING) else None
    return state, job.hostname, str(job.runningTime), millify(job.maxVmSize)


# ---------------------------------------------------------------------------
# Judging a done job
# ---------------------------------------------------------------------------


def judge(job, slurm=None):
    """(status, stats) of a done job; slurm is its sacct outcome, if any."""
    for _ in range(5):  # the scheduler can report a job done before its files show up
        if os.path.exists(job["outfile"]):
            break
        time.sleep(1)
    found, stats = set(), {}
    if slurm is not None and slurm.get("state") != "COMPLETED":
        found.add("SLURM_{0}({1})".format(slurm.get("state") or "UNKNOWN", slurm.get("exit", "")))
    if not os.path.exists(job["outfile"]):
        found.add(NO_OUTPUT)
        return ", ".join(sorted(found)), ""
    text = common.tail_text([job["outfile"], job["errfile"]], TAIL_BYTES)
    passed = bool(PASS_TOKEN.search(text))
    for token, name in FAIL_TOKENS.items():
        if re.search(token, text):
            found.add(name)
    for line in reversed(text.splitlines()):
        for name, token in STATS_TO_PULL.items():
            if name not in stats:
                m = re.search(token, line)
                if m:
                    stats[name] = m.group(1).strip()
        if len(stats) == len(STATS_TO_PULL):
            break
    stat = "\t".join("{0}={1}".format(k, stats[k]) for k in STATS_TO_PULL if k in stats)
    if found:
        return ", ".join(sorted(found)), stat
    if passed:
        return PASSED, stat
    if os.path.exists(job["errfile"]) and os.path.getsize(job["errfile"]) > 0:
        return ERR_FILE, stat
    return NO_ERROR, stat


def is_failure(status):
    return status not in (PASSED, NO_ERROR, RUNNING, WAITING)


# ---------------------------------------------------------------------------
# One poll
# ---------------------------------------------------------------------------


def status(logfile, run_dir, job_manager):
    """[(job, status, node, time, mem, stat)] for every job of a launch log, and
    the parse warnings. Done jobs are judged once and cached next to the log."""
    jobs, warnings = parse_logfile(logfile, run_dir)
    cache_file = re.sub("sim_log", "node_details", logfile)
    try:
        cache = json.load(open(cache_file))
    except (OSError, ValueError):
        cache = {}

    open_jobs = [j for j in jobs if "final" not in cache.get(j["id"], {})]
    live, outcomes = {}, None
    if job_manager == "squeue" and open_jobs:
        live = slurm_live()
        gone = [j["id"] for j in open_jobs if j["id"] not in live]
        outcomes = common.slurm_outcomes(gone) if gone else {}
        if outcomes is None:
            warnings.append("no sacct: done jobs are judged by their output only")

    for j in open_jobs:
        entry = cache.setdefault(j["id"], {})
        state, mem = None, ""
        if job_manager == "squeue":
            if j["id"] in live:
                state, entry["node"], entry["time"] = live[j["id"]]
            elif outcomes is not None:
                o = outcomes.get(j["id"])
                if o is None:
                    entry["misses"] = entry.get("misses", 0) + 1
                    if entry["misses"] < SACCT_MISSES:
                        state = RUNNING
                elif o["state"] in SLURM_ACTIVE:
                    state = WAITING if o["state"] == "PENDING" else RUNNING
        elif job_manager == "qstat":
            state, entry["node"], entry["time"] = qstat_state(j["id"])
        else:
            state, node, elapsed, mem = procman_state(j["id"])
            # procman -S clears the launcher's pickle before the procman it spawned saves its
            # own, so right after a launch no pickle may hold a job that has not started yet
            if state is None and not os.path.exists(j["outfile"]):
                state = UNKNOWN
            if state == UNKNOWN:
                entry["misses"] = entry.get("misses", 0) + 1
                if entry["misses"] < SACCT_MISSES:
                    continue
                state = None
            else:
                entry["misses"], entry["node"], entry["time"] = 0, node, elapsed
        if state:
            entry["live"] = state
            continue
        slurm = outcomes.get(j["id"]) if outcomes else None
        st, stat = judge(j, slurm)
        if slurm and slurm.get("maxrss_kb"):
            mem = millify(slurm["maxrss_kb"] * 1000)
        entry["final"] = {"status": st, "stat": stat, "mem": mem}

    try:
        json.dump(cache, open(cache_file, "w"))
    except OSError as e:
        warnings.append("cannot write {0}: {1}".format(cache_file, e))

    rows = []
    for j in jobs:
        e = cache.get(j["id"], {})
        f = e.get("final")
        rows.append((j, f["status"] if f else e.get("live", WAITING), e.get("node", ""),
                     e.get("time", ""), f["mem"] if f else "", f["stat"] if f else ""))
    return rows, warnings


def write_failed_log(logfile, rows, num_lines):
    """The failed jobs' rows, .o tails and .e files; returns the log's path."""
    path = os.path.join(os.path.dirname(logfile), "failed_job_log_" + os.path.basename(logfile))
    with open(path, "w") as f:
        failed = [r for r in rows if is_failure(r[1])]
        f.write("".join(format_row(*r) + "\n" for r in failed) + "\n")
        for j, st, *_ in failed:
            f.write("*" * 58 + "\n{0}-{1}--{2}. Status={3}\n".format(j["app"], j["args"], j["config"], st))
            f.write("Last {0} lines of {1}\n------------------\n".format(num_lines, j["outfile"]))
            if os.path.exists(j["outfile"]):
                f.write("\n".join(common.tail_text([j["outfile"]], 64 << 10).splitlines()[-num_lines:]) + "\n")
            else:
                f.write("(missing)\n")
            f.write("------------------\n\nContents of {0}\n------------------\n".format(j["errfile"]))
            f.write((common.tail_text([j["errfile"]], 64 << 10) if os.path.exists(j["errfile"]) else "(missing)") + "\n")
            f.write("------------------\n" + "*" * 58 + "\n")
    return path


def format_row(job, status, node, elapsed, mem, stat):
    return ROW_STRING.format(jobId=job["id"], exec_node=node or "UNKNOWN", app=job["app"],
                             args=job["args"], version=job["version"], config=job["config"],
                             running_time=elapsed or "UNKNOWN", mem_used=mem or "UNKNOWN",
                             status=status, stat=stat)


def header(job_manager):
    return ROW_STRING.format(jobId=job_manager + ".id", exec_node="Node", app="App", args="AppArgs",
                             version="Version", config="Config", running_time="RunningTime",
                             mem_used="Mem", status="JobStatus", stat="Basic GPGPU-Sim Stats")


def resolve_run_dir(run_dir):
    run_dir = common.dir_option_test(
        run_dir, this_directory + ("../../sim_run_%s/" % common.get_cuda_version(this_directory)), this_directory)
    if not os.path.isdir(run_dir):
        exit(run_dir + " does not exist - specify the run directory where the benchmark/config dirs exist")
    return run_dir


def add_common_options(parser):
    parser.add_option("-l", "--logfile", dest="logfile", default="",
                      help="The logfile the status is based on. By default the latest launch's; "
                      '"all" for every launch in the logfile directory.')
    parser.add_option("-r", "--run_dir", dest="run_dir", default="",
                      help="The directory where the benchmark/config directories exist.")
    parser.add_option("-N", "--sim_name", dest="sim_name", default="",
                      help="The latest launch of this run_simulations.py -N name.")
    parser.add_option("-j", "--job_manager", dest="job_manager", default=None,
                      help="squeue (Slurm), qstat (Torque) or procman. Default: whichever is installed.")


def main():
    parser = OptionParser()
    add_common_options(parser)
    parser.add_option("-n", "--num_lines", dest="num_lines", default="10",
                      help="Number of lines of a failed job's .o file to log. Default is 10")
    options, _ = parser.parse_args()
    job_manager = pick_job_manager(options.job_manager)
    run_dir = resolve_run_dir(options.run_dir.strip())
    logfiles = find_logfiles(options.logfile.strip(), options.sim_name.strip())
    print("Using logfiles " + str(logfiles))
    for logfile in logfiles:
        rows, warnings = status(logfile, run_dir, job_manager)
        h = header(job_manager)
        print(h + "\n" + "-" * len(h))
        for r in rows:
            print(format_row(*r))
        print("-" * len(h))
        for w in warnings:
            print("WARNING " + w)
        if any(is_failure(r[1]) for r in rows):
            print("failed job log written to " + write_failed_log(logfile, rows, int(options.num_lines)))


if __name__ == "__main__":
    main()
