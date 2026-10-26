#!/bin/bash
# Main CI PTX Simulation Script
#
# This script implements the PTX-Simulation job from main.yml.
# It can be run locally to reproduce the CI flow, or called with specific stages from CI.
#
# Usage:
#   ./main-ptx.sh              # Run everything
#   ./main-ptx.sh build        # Run specific stage
#   ./main-ptx.sh build-apps
#   ./main-ptx.sh simulate
#
# Environment variables (set automatically in CI, or manually for local runs):
#   BRANCH_NAME             - Git branch name
#   GITHUB_REPOSITORY       - Repository in owner/repo format
#   GITHUB_EVENT_NAME       - Event type (push, pull_request)
#   PR_HEAD_REPO_FULL_NAME  - For PRs: head repo in owner/repo format
#   GITHUB_RUN_NUMBER       - CI run number
#   GITHUB_RUN_ATTEMPT      - CI run attempt

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

#==============================================================================
# CONFIGURATION
#==============================================================================

# Build settings
USE_SRUN=true
BUILD_JOBS=20

# PTX configs (no trace paths needed)
CONFIGS="QV100-PTX,A100-PTX"
# The lab's extracted all.gpgpu-sim-app-data.tgz, shared with the functional CI
APPDATA_DIRS="/scratch/tgrogers-disk01/a/common/for-sharing/gpgpu-sim-appdata/data_dirs"

# Suites simulated, by tier as in main-sass.sh: the short ubench args on PRs, the
# full ones on dev. MAKE_TARGETS builds both (the _CI suite is a define-all-apps
# entry with short args, not a make target).
CI_TIER="${CI_TIER:-smoke}"
case "$CI_TIER" in
    smoke) BENCHMARKS="rodinia_2.0-ft,GPU_Microbenchmark_CI" ;;
    full)  BENCHMARKS="rodinia_2.0-ft,GPU_Microbenchmark" ;;
    *)
        echo "CI_TIER must be smoke or full, not '$CI_TIER'" >&2
        exit 1
        ;;
esac
MAKE_TARGETS="rodinia_2.0-ft GPU_Microbenchmark"

# Job naming
JOB_NAME="short-ptx-$(ci_job_name)"

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

stage_build_apps() {
    log_section "Building GPU Applications"
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-simulator/setup_environment.sh

    rm -rf ./gpu-app-collection
    git clone --quiet --recurse-submodules -b dev git@github.com:accel-sim/gpu-app-collection.git
    source ./gpu-app-collection/src/setup_environment > /dev/null 2>&1

    local make_targets="$MAKE_TARGETS"

    if [ "$USE_SRUN" = "true" ]; then
        srun --time=8:00:00 -c"$BUILD_JOBS" make $make_targets -j"$BUILD_JOBS" -C ./gpu-app-collection/src
    else
        make $make_targets -j"$BUILD_JOBS" -C ./gpu-app-collection/src
    fi

    # get_regression_data.sh downloads only when data_dirs is missing, and
    # exits 0 on a truncated download
    [ -d "$APPDATA_DIRS/cuda/rodinia/2.0-ft" ] || { log_error "no rodinia_2.0-ft data under $APPDATA_DIRS"; exit 1; }
    ln -sfn "$APPDATA_DIRS" ./gpu-app-collection/data_dirs
    log_info "Applications built successfully"
}

stage_simulate() {
    log_section "Running PTX Simulations"
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-simulator/setup_environment.sh
    source ./gpu-app-collection/src/setup_environment

    ./util/job_launching/run_simulations.py \
        -B "$BENCHMARKS" \
        -C "$CONFIGS" \
        -N "$JOB_NAME"

    monitor_simulations "$JOB_NAME" 12 300 "stats-per-app-ptx.csv"
}

# Run all stages
run_all() {
    stage_setup
    stage_build
    stage_build_apps
    stage_simulate
}

#==============================================================================
# MAIN DISPATCH
#==============================================================================

show_usage() {
    echo "Usage: $0 {setup|build|build-apps|simulate|all}"
    echo ""
    echo "Stages:"
    echo "  setup      - Clone and setup env-setup repository"
    echo "  build      - Clone gpgpu-sim and build accel-sim"
    echo "  build-apps - Clone and build gpu-app-collection"
    echo "  simulate   - Run PTX simulations for QV100, A100, H200"
    echo "  all        - Run all stages in order"
}

case "${1:-all}" in
    setup)      stage_setup ;;
    build)      stage_build ;;
    build-apps) stage_build_apps ;;
    simulate)   stage_simulate ;;
    all)        run_all ;;
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
