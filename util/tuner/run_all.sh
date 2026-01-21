#!/bin/bash

# THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
source ./gpu-app-collection-partial/src/setup_environment
SCRIPT_DIR="./gpu-app-collection-partial/src/cuda/GPU_Microbenchmark/"
echo "Running make in $SCRIPT_DIR"
make -C "$SCRIPT_DIR" -j || { echo "make failed"; exit 1; }

cd ${SCRIPT_DIR}/bin/

# List of configuration benchmarks that output lines starting with "-"
# These are used by tuner.py to generate Accel-Sim configuration

# System config
echo "running system_config"
./system_config
echo "/////////////////////////////////"

# Core config
echo "running core_config"
./core_config
echo "/////////////////////////////////"

echo "running config_dpu"
./config_dpu --blocks 1
echo "/////////////////////////////////"

echo "running config_fpu"
./config_fpu --blocks 1
echo "/////////////////////////////////"

echo "running config_int"
./config_int --blocks 1
echo "/////////////////////////////////"

echo "running config_sfu"
./config_sfu --blocks 1
echo "/////////////////////////////////"

echo "running config_tensor"
./config_tensor --blocks 1
echo "/////////////////////////////////"

echo "running config_udp"
./config_udp --blocks 1
echo "/////////////////////////////////"

echo "running regfile_bw"
./regfile_bw
echo "/////////////////////////////////"

# L1 cache config
echo "running l1_config"
./l1_config
echo "/////////////////////////////////"

echo "running l1_lat with args: --blocks 1"
./l1_lat --blocks 1
echo "/////////////////////////////////"

# L2 cache config
echo "running l2_config"
./l2_config
echo "/////////////////////////////////"

echo "running l2_copy_engine"
./l2_copy_engine
echo "/////////////////////////////////"

echo "running l2_lat"
./l2_lat
echo "/////////////////////////////////"

# Memory config
echo "running mem_config"
./mem_config
echo "/////////////////////////////////"

echo "running mem_lat"
./mem_lat
echo "/////////////////////////////////"

# Shared memory config
echo "running shd_config"
./shd_config
echo "/////////////////////////////////"

echo "running shared_lat with args: --blocks 1"
./shared_lat --blocks 1
echo "/////////////////////////////////"

# Kernel latency
echo "running kernel_lat"
./kernel_lat
echo "/////////////////////////////////"
