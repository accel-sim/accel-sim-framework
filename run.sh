#!/bin/bash

# Save directory paths
SCRIPT_DIR=$(dirname -- "$0")
ORIGINAL_PWD=$(pwd)

# Navigate to script directory and setup environment
cd "$SCRIPT_DIR"
source gpu-simulator/setup_environment.sh

# Run QV100 simulations
python3 ./util/job_launching/run_simulations.py \
    -B rodinia_2.0-ft,rodinia-3.1,GPU_Microbenchmark \
    -C QV100-SASS,QV100-SASS-SIMPLE_DRAM,QV100-SASS-SIMPLE_DRAM-CORRECT_V100 \
    -T /scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/QV100/traces/device-6/12.8/ \
    -N v100-$(date +%Y%m%d_%H%M%S)

# Run A100 simulations
# python3 ./util/job_launching/run_simulations.py \
#     -B rodinia_2.0-ft,GPU_Microbenchmark \
#     -C A100-SASS,A100-SASS-SIMPLE_DRAM \
#     -T /scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/A100/hw_run/traces/device-0/12.8/ \
#     -N a100-$(date +%Y%m%d_%H%M%S)

# # Run H100 simulations
# python3 ./util/job_launching/run_simulations.py \
#     -B rodinia_2.0-ft,GPU_Microbenchmark \
#     -C H100-SASS,H100-SASS-SIMPLE_DRAM \
#     -T /scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/H100/hw_run/traces/device-0/12.8/ \
#     -N h100-$(date +%Y%m%d_%H%M%S)

# Return to original directory
cd "$ORIGINAL_PWD"
