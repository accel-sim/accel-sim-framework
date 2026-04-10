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
#   ./main-tracer.sh hw-stats     # Generate HW stats
#   ./main-tracer.sh test         # Test new traces
#
# Environment variables (set automatically in CI, or manually for local runs):
#   BRANCH_NAME             - Git branch name
#   GITHUB_REPOSITORY       - Repository in owner/repo format
#   GITHUB_EVENT_NAME       - Event type (push, pull_request)
#   PR_HEAD_REPO_FULL_NAME  - For PRs: head repo in owner/repo format

set -e

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
    srun --job-name=gpu-lock --dependency=singleton --partition=tgrogers-dgx --time=02:00:00 -- \
        ./util/tracer_nvbit/run_hw_trace.py -B "$BENCHMARKS" -D "$GPU_DEVICE"

    log_info "Traces generated successfully"
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
        -N "$JOB_NAME"

    monitor_simulations "$JOB_NAME" 12 300 "rodinia-stats-per-app.csv"
}

# Run all stages
run_all() {
    stage_setup
    stage_build
    stage_build_tracer
    stage_build_apps
    stage_trace
    stage_hw_stats
    stage_test
}

#==============================================================================
# MAIN DISPATCH
#==============================================================================

show_usage() {
    echo "Usage: $0 {setup|build|build-tracer|build-apps|trace|hw-stats|test|all}"
    echo ""
    echo "Stages:"
    echo "  setup        - Clone and setup env-setup repository"
    echo "  build        - Clone gpgpu-sim and build accel-sim"
    echo "  build-tracer - Build the NVBit tracer tool"
    echo "  build-apps   - Clone and build gpu-app-collection"
    echo "  trace        - Generate traces on GPU (uses srun)"
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
