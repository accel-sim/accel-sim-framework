#!/bin/bash
set -e

echo "Starting H200 trace generation..."

# Setup environment
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8/
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

# Build Tracer
echo "Building tracer..."
./util/tracer_nvbit/install_nvbit.sh
make clean -C ./util/tracer_nvbit/
make -C ./util/tracer_nvbit/

# Build applications
echo "Building applications..."
rm -rf ./gpu-app-collection/
git clone --quiet --recurse-submodules https://github.com/accel-sim/gpu-app-collection.git
source ./gpu-app-collection/src/setup_environment > /dev/null 2>&1
make -C ./gpu-app-collection/src -j64 rodinia_2.0-ft rodinia-3.1 GPU_Microbenchmark cutlass_mini GPU_Microbenchmark_gmma
# create dummy data_dirs
ln -s /purdue/tgrghci/gpu-app-collection/data_dirs ./gpu-app-collection/data_dirs

# Generate traces
echo "Generating traces..."
source ./gpu-app-collection/src/setup_environment
rm -rf ./hw_run/
./util/tracer_nvbit/run_hw_trace.py -B rodinia_2.0-ft,rodinia-3.1,GPU_Microbenchmark -D 7
ALLOW_REG_VAL_TRACING=1 ./util/tracer_nvbit/run_hw_trace.py -B GPU_Microbenchmark_TMA,cutlass_tma_small -D 7 --spinlock_handling fast_forward

# Generate hardware stats
echo "Generating hardware stats..."
./util/hw_stats/run_hw.py -B rodinia_2.0-ft,rodinia-3.1,GPU_Microbenchmark -D 7
./util/hw_stats/run_hw.py -B GPU_Microbenchmark_TMA,cutlass_tma_small -D 7
echo "H200 trace generation completed!"
