#!/bin/bash
# Cancel this user's queued/running Slurm jobs tagged $ACCELSIM_SLURM_COMMENT
# (exactly, or as the prefix of <tag>:<job>).
set -euo pipefail
: "${ACCELSIM_SLURM_COMMENT:?ACCELSIM_SLURM_COMMENT must be set}"
jobs=$(squeue -h -u "$(id -un)" -o '%i %k' | awk -v t="$ACCELSIM_SLURM_COMMENT" '$2 == t || index($2, t ":") == 1 {print $1}')
echo "cancelling: ${jobs:-none}"
[ -z "$jobs" ] || scancel $jobs
