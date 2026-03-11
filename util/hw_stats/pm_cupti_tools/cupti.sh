#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <program> [args...]"
    exit 1
fi

METRICS=""

METRICS="sm__cycles_elapsed.avg"
METRICS="$METRICS,sm__inst_executed.sum"

export PM_SAMPLING_MAX_SAMPLES=80000
export INJECTION_METRICS=$METRICS
export INJECTION_KERNEL_COUNT=10
export PM_SAMPLING_HW_BUFFER_BYTES=9388608000
export PM_SAMPLING_INTERVAL_SYSCLK=3000
# export PM_SAMPLING_CSV_PATH=$PWD/pm_samples.csv
# export LD_LIBRARY_PATH=/usr/local/cuda-12.8/extras/CUPTI/lib64:$LD_LIBRARY_PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_INJECTION64_PATH="$SCRIPT_DIR/build/libpmsampling_injection.so"

exec "$@"

