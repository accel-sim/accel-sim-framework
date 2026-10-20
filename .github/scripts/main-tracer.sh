#!/bin/bash
# Main CI Tracer Tool Script
#
# This script implements the Tracer-Tool job from main.yml.
# It builds the tracer, generates traces on the local GPU, and tests them.
#
# Usage:
#   ./main-tracer.sh              # Run everything
#   ./main-tracer.sh build        # Build accel-sim
#   ./main-tracer.sh build-tracer # Build tracer tool
#   ./main-tracer.sh build-apps   # Build applications
#   ./main-tracer.sh trace        # Generate traces
#   ./main-tracer.sh trace-parallel # Trace again on several GPUs, compare
#   ./main-tracer.sh hw-stats     # Generate HW stats
#   ./main-tracer.sh test         # Test new traces
#
# Environment variables (set automatically in CI, or manually for local runs):
#   BRANCH_NAME             - Git branch name
#   GITHUB_REPOSITORY       - Repository in owner/repo format
#   GITHUB_EVENT_NAME       - Event type (push, pull_request)
#   PR_HEAD_REPO_FULL_NAME  - For PRs: head repo in owner/repo format

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

#==============================================================================
# CONFIGURATION
#==============================================================================

# Build settings
USE_SRUN=false
BUILD_JOBS=20

# Tracer settings
GPU_DEVICE=7  # GPU device to use for tracing
PARALLEL_GPU_DEVICES="3,7"  # GPUs for the run_hw_trace.py worker pool test
BENCHMARKS="rodinia_2.0-ft"
CONFIG="QV100-SASS"
DATA_DIRS_PATH="/home/tgrogers-raid/a/common/data_dirs"

# Job naming
JOB_NAME="rodinia_2.0-ft-$$"

#==============================================================================
# STAGES
#==============================================================================

stage_setup() {
    setup_environment
}

stage_build() {
    log_section "Building Accel-Sim"
    source ./env-setup/12.8_env_setup.sh
    clone_gpgpusim
    build_accelsim "$USE_SRUN" "$BUILD_JOBS"
}

stage_build_tracer() {
    log_section "Building Tracer Tool"
    source ./env-setup/12.8_env_setup.sh

    ./util/tracer_nvbit/install_nvbit.sh
    make clean -C ./util/tracer_nvbit/
    make -C ./util/tracer_nvbit/

    log_info "Tracer built successfully"
}

stage_build_apps() {
    log_section "Building GPU Applications"
    source ./env-setup/12.8_env_setup.sh

    rm -rf ./gpu-app-collection/
    git clone --quiet git@github.com:accel-sim/gpu-app-collection.git
    source ./gpu-app-collection/src/setup_environment > /dev/null 2>&1

    # Link data directories
    ln -s "$DATA_DIRS_PATH" ./gpu-app-collection/

    make -C ./gpu-app-collection/src "$BENCHMARKS"
    log_info "Applications built successfully"
}

stage_trace() {
    log_section "Generating Traces"
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-app-collection/src/setup_environment

    rm -rf ./hw_run/
    # nvbit-trace.sh also removes NVBit's /tmp files on the GPU node
    srun --job-name=gpu-lock --dependency=singleton --partition=tgrogers-dgx --time=02:00:00 -- \
        "$SCRIPT_DIR/nvbit-trace.sh" -B "$BENCHMARKS" -D "$GPU_DEVICE"

    log_info "Traces generated successfully"
}

# The apps of stage_trace again, spread over PARALLEL_GPU_DEVICES by
# run_hw_trace.py's GPU tracing workers and post-processed by two CPU workers.
# The traces must match stage_trace's (compare-traces.py), and every listed GPU
# must have traced an app.
stage_trace_parallel() {
    log_section "Tracing on GPUs $PARALLEL_GPU_DEVICES (worker pools)"
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-app-collection/src/setup_environment

    rm -rf ./hw_run_parallel/
    srun --job-name=gpu-lock --dependency=singleton --partition=tgrogers-dgx --time=02:00:00 --cpus-per-task=8 -- \
        "$SCRIPT_DIR/nvbit-trace.sh" -B "$BENCHMARKS" -D "$PARALLEL_GPU_DEVICES" \
        --cpu_workers 2 -j 4 --output_name hw_run_parallel

    local serial=(./hw_run/traces/device-"$GPU_DEVICE"/*/)
    local parallel=(./hw_run_parallel/traces/device-"${PARALLEL_GPU_DEVICES%%,*}"/*/)
    python3 "$SCRIPT_DIR/compare-traces.py" "${serial[0]}" "${parallel[0]}"

    local used
    used=$(find ./hw_run_parallel -name traced_on_gpu.txt -exec tail -n 1 {} \; | cut -d, -f1 | sort -n -u | paste -sd, -)
    if [ "$used" != "$(echo "$PARALLEL_GPU_DEVICES" | tr ',' '\n' | sort -n -u | paste -sd, -)" ]; then
        log_error "Apps were traced on GPUs '$used', expected every GPU of $PARALLEL_GPU_DEVICES"
        exit 1
    fi

    log_info "Worker pool traces match the single-GPU traces (traced on GPUs $used)"
}

stage_hw_stats() {
    log_section "Generating Hardware Stats"
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-app-collection/src/setup_environment

    srun --job-name=gpu-lock --dependency=singleton --partition=tgrogers-dgx --time=02:00:00 -- \
        ./util/hw_stats/run_hw.py -B "$BENCHMARKS" -D "$GPU_DEVICE"

    log_info "HW stats generated successfully"
}

stage_test() {
    log_section "Testing New Traces"
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-simulator/setup_environment.sh

    ./util/job_launching/run_simulations.py \
        -B "$BENCHMARKS" \
        -C "$CONFIG" \
        -T ./hw_run/traces/ \
        -N "$JOB_NAME" \
        "${SIM_PER_KERNEL[@]}"

    monitor_simulations "$JOB_NAME" 12 300 "rodinia-stats-per-app.csv" -v
}

# Run all stages
run_all() {
    stage_setup
    stage_build
    stage_build_tracer
    stage_build_apps
    stage_trace
    stage_trace_parallel
    stage_hw_stats
    stage_test
}

#==============================================================================
# MAIN DISPATCH
#==============================================================================

show_usage() {
    echo "Usage: $0 {setup|build|build-tracer|build-apps|trace|trace-parallel|hw-stats|test|all}"
    echo ""
    echo "Stages:"
    echo "  setup        - Clone and setup env-setup repository"
    echo "  build        - Clone gpgpu-sim and build accel-sim"
    echo "  build-tracer - Build the NVBit tracer tool"
    echo "  build-apps   - Clone and build gpu-app-collection"
    echo "  trace        - Generate traces on GPU (uses srun)"
    echo "  trace-parallel - Trace on several GPUs, compare with trace (uses srun)"
    echo "  hw-stats     - Generate hardware stats (uses srun)"
    echo "  test         - Test traces with simulation"
    echo "  all          - Run all stages in order"
}

case "${1:-all}" in
    setup)        stage_setup ;;
    build)        stage_build ;;
    build-tracer) stage_build_tracer ;;
    build-apps)   stage_build_apps ;;
    trace)        stage_trace ;;
    trace-parallel) stage_trace_parallel ;;
    hw-stats)     stage_hw_stats ;;
    test)         stage_test ;;
    all)          run_all ;;
    -h|--help|help)
        show_usage
        ;;
    *)
        echo "Unknown stage: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
