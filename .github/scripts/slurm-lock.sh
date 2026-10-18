#!/bin/bash
# Hold a lock shared by every repo's CI on this cluster, through Slurm: a
# do-nothing job named NAME with --dependency=singleton, which Slurm runs for
# only one job of that name per user (all runners submit as the CI user).
#
#   slurm-lock.sh NAME TIME     return once this run's holder is RUNNING
#
# The holder carries the run's tag (ACCELSIM_SLURM_COMMENT), so a later job of
# the same run finds and reuses it, and cleanup-slurm's scancel by that tag
# releases it when the run ends. TIME (sbatch --time) covers a dead runner.
set -euo pipefail
name=${1:?usage: $0 NAME TIME}
time=${2:?usage: $0 NAME TIME}
tag=${ACCELSIM_SLURM_COMMENT:?ACCELSIM_SLURM_COMMENT not set: the holder must carry the run tag}

holder() { squeue -h -u "$(id -un)" -n "$name" -o '%i %k' | awk -v t="$tag" '$2 == t {print $1; exit}'; }

id=$(holder)
if [ -z "$id" ]; then
    id=$(sbatch --parsable -J "$name" --dependency=singleton --time="$time" \
        --comment="$tag" -p batch -c 1 --mem=100M -o /dev/null --wrap 'sleep infinity')
    echo "submitted lock holder $id ($name)"
else
    echo "reusing this run's lock holder $id ($name)"
fi
last=0
while :; do
    state=$(squeue -h -j "$id" -o '%T' 2>/dev/null || true)
    case $state in
        RUNNING) echo "holding $name (job $id)"; exit 0 ;;
        PENDING|CONFIGURING)
            now=$(date +%s)
            if [ $((now - last)) -ge 600 ]; then
                echo "waiting for $name: held by $(squeue -h -u "$(id -un)" -n "$name" -t RUNNING -o '%i %k' | head -1)"
                last=$now
            fi
            sleep "${SLURM_LOCK_POLL_SEC:-30}" ;;
        *) echo "::error::lock holder $id ($name) is ${state:-gone}, not pending or running"; exit 1 ;;
    esac
done
