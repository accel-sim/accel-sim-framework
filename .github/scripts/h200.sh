#!/bin/bash
# Traces and hardware stats of suites $1 (comma-separated) into ./hw_run, run
# on the remote H200 by weekly.sh trace-h200.
set -eo pipefail

suites="${1:?usage: h200.sh <comma-separated suites>}"
device=7

export CUDA_INSTALL_PATH=/usr/local/cuda-12.8/
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

echo "Building the tracer..."
./util/tracer_nvbit/install_nvbit.sh
make clean -C ./util/tracer_nvbit/
make -C ./util/tracer_nvbit/

echo "Building the applications..."
rm -rf ./gpu-app-collection/
git clone --quiet --recurse-submodules https://github.com/accel-sim/gpu-app-collection.git
source ./gpu-app-collection/src/setup_environment > /dev/null 2>&1
# GPU_Microbenchmark_TMA and cutlass_tma_small are built by the last two targets
make -C ./gpu-app-collection/src -j64 rodinia_2.0-ft rodinia-3.1 GPU_Microbenchmark \
    GPU_Microbenchmark_gmma cutlass_mini
ln -s /purdue/tgrghci/gpu-app-collection/data_dirs ./gpu-app-collection/data_dirs

echo "Tracing $suites..."
source ./gpu-app-collection/src/setup_environment
rm -rf ./hw_run/
./util/tracer_nvbit/run_hw_trace.py -B "$suites" -D "$device"

echo "Collecting hardware stats of $suites..."
./util/hw_stats/run_hw.py -B "$suites" -D "$device"
