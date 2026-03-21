#!/bin/bash
# Cancel Slurm jobs by simulation name
# Usage: cancel-slurm-jobs.sh <sim-name>
#
# Finds the most recent logfile matching the simulation name and cancels
# all jobs listed in it.

set -e

SIM_NAME="$1"
if [ -z "$SIM_NAME" ]; then
    echo "Usage: $0 <sim-name>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$SCRIPT_DIR/../../util/job_launching/logfiles"

# Find the most recent logfile matching the simulation name
LOGFILE=$(ls -t "$LOGDIR"/sim_log."$SIM_NAME".* 2>/dev/null | head -1)

if [ -f "$LOGFILE" ]; then
    echo "Found logfile: $LOGFILE"
    # Extract job IDs (2nd column) and cancel them
    awk '{print $2}' "$LOGFILE" | while read jobid; do
        if [ -n "$jobid" ] && [[ "$jobid" =~ ^[0-9]+$ ]]; then
            echo "Cancelling job $jobid"
            scancel "$jobid" 2>/dev/null || true
        fi
    done
else
    echo "No logfile found for simulation: $SIM_NAME"
fi
