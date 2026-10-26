#!/usr/bin/env python3
"""Poll a run_simulations.py launch until every job is done; exit 1 if any failed.

    monitor_func_test.py -N <launch name> [-T hours] [-S seconds] [-v] [-s stats.csv]
                         [--allow-oom] [-I] [-K]

Each poll prints one summary line; at the end, the failed jobs' rows. -v also
prints every job's row each poll. What counts as a failure is job_status.py's: with Slurm, any job sacct does not show
COMPLETED, besides the output checks every scheduler gets.
"""

from optparse import OptionParser
import os
import subprocess
import time

import common
import job_status


def write_statsfile(options):
    with open(options.statsfile, "w+") as f:
        cmd = [os.path.join(job_status.this_directory, "get_stats.py"), "-R", "-l", options.logfile,
               "-N", options.sim_name, "-A"]
        if subprocess.call(cmd, stdout=f, stderr=f) != 0:
            print("Error Launching get_stats.py")
        if options.verbose:
            f.seek(0)
            print(f.read())


def only_oom_failures(job_ids):
    """--allow-oom: True if Slurm shows every failed job was killed for memory
    (common.oom_killed); one sacct call for the failed jobs only."""
    outcomes = common.slurm_outcomes(job_ids)
    if outcomes is None:
        print("--allow-oom: no sacct (not Slurm); failures stand")
        return False
    oom = [j for j in job_ids if common.oom_killed(outcomes.get(j))]
    other = [j for j in job_ids if j not in oom]
    if oom:
        print("OOM-killed, not a failure ({0}): {1}".format(len(oom), " ".join(oom)))
    if other:
        print("Failed, not OOM ({0}): {1}".format(len(other), " ".join(other)))
    return not other


def kill_unfinished(job_manager, rows):
    ids = [r[0]["id"] for r in rows if r[1] in (job_status.RUNNING, job_status.WAITING)]
    if not ids:
        return
    print("Killing {0} unfinished jobs".format(len(ids)))
    if job_manager == "squeue":
        cmds = [["scancel"] + ids[i : i + 500] for i in range(0, len(ids), 500)]
    elif job_manager == "qstat":
        cmds = [["qdel", i] for i in ids]
    else:
        cmds = [[os.path.join(job_status.this_directory, "procman.py"), "-k"]]
    for cmd in cmds:
        if subprocess.call(cmd) != 0:
            print("WARNING error calling " + cmd[0])


def print_rows(job_manager, rows):
    h = job_status.header(job_manager)
    print(h + "\n" + "-" * len(h))
    for r in rows:
        print(job_status.format_row(*r))
    print("-" * len(h))


def main():
    parser = OptionParser()
    job_status.add_common_options(parser)
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="Print every job's status each poll, and the stats file at the end")
    parser.add_option("-s", "--statsfile", dest="statsfile", default="",
                      help="Where get_stats.py's output goes when every job passed")
    parser.add_option("-S", "--sleep_time", dest="sleep_time", default="30",
                      help="Time to sleep in (s) - default is 30.")
    parser.add_option("-I", "--ignore_failures", dest="ignore_failures", action="store_true",
                      help="If some of the runs have errors - do not return an error code.")
    parser.add_option("-T", "--timeout", dest="timeout", default="99999",
                      help="Maximum number of hours to run the monitor for. If time is exceeded, exit with error code.")
    parser.add_option("-K", "--killwhentimedout", dest="killwhentimedout", action="store_true",
                      help="Kill all the jobs still running if the monitor times out.")
    parser.add_option("--allow-oom", dest="allow_oom", action="store_true",
                      help="With Slurm: at the end, if every failed job was killed for memory "
                      "(common.oom_killed on sacct's state, exit code and MaxRSS), exit 0")
    options, _ = parser.parse_args()
    options.logfile, options.sim_name = options.logfile.strip(), options.sim_name.strip()
    job_manager = job_status.pick_job_manager(options.job_manager)
    run_dir = job_status.resolve_run_dir(options.run_dir.strip())
    logfiles = job_status.find_logfiles(options.logfile, options.sim_name)
    if len(logfiles) != 1:
        exit("monitor_func_test.py follows one launch, not {0} logfiles".format(len(logfiles)))
    logfile = logfiles[0]
    deadline = time.time() + float(options.timeout) * 3600

    while True:
        rows, warnings = job_status.status(logfile, run_dir, job_manager)
        if options.verbose:
            print_rows(job_manager, rows)
            for w in warnings:
                print("WARNING " + w)
        n = {k: 0 for k in ("passed", "no_err", "failed", "running", "waiting")}
        for _, st, *_ in rows:
            n[{job_status.PASSED: "passed", job_status.NO_ERROR: "no_err", job_status.RUNNING: "running",
               job_status.WAITING: "waiting"}.get(st, "failed")] += 1
        print("Passed:{passed}/{t}, No error:{no_err}/{t}, Failed/Error:{failed}/{t}, "
              "Running:{running}/{t}, Waiting:{waiting}/{t}".format(t=len(rows), **n))
        failed = [r for r in rows if job_status.is_failure(r[1])]
        timed_out = time.time() >= deadline
        if n["running"] + n["waiting"] == 0 or timed_out:
            if not options.verbose:
                for w in warnings:
                    print("WARNING " + w)
            if failed:
                print_rows(job_manager, failed)
                print("Failed jobs' output: " + job_status.write_failed_log(logfile, rows, 10))
            if timed_out and n["running"] + n["waiting"]:
                print("Monitor has timed-out")
                if options.killwhentimedout:
                    kill_unfinished(job_manager, rows)
                exit(1)
            print("All {0} Tests Done.".format(len(rows)))
            print("Something did not pass." if failed else "Congratulations! All Tests Pass!")
            ok = (not failed or options.ignore_failures
                  or (options.allow_oom and only_oom_failures([r[0]["id"] for r in failed])))
            if ok and options.statsfile:
                write_statsfile(options)
            exit(0 if ok else 1)
        print("Sleeping for {0}s".format(options.sleep_time))
        time.sleep(int(options.sleep_time))


if __name__ == "__main__":
    main()
