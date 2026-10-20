#!/bin/bash
# Main CI SASS-Simulation-Docker Script
#
# This script implements the SASS-Simulation-Docker job from main.yml: the
# short QV100 SASS regression (what short-tests.sh runs) built and run inside
# the accel-sim-framework Docker image, to check the image's toolchain. The
# image runs under apptainer on a Slurm batch node (on-cluster.sh).
#
# Usage:
#   ./main-docker.sh              # Run everything
#   ./main-docker.sh setup        # On the runner: clone gpgpu-sim, get traces
#   ./main-docker.sh run          # Build and simulate in the image (via Slurm)
#   ./main-docker.sh in-image     # What `run` executes inside the image
#
# Environment variables (set automatically in CI, or manually for local runs):
#   BRANCH_NAME             - Git branch name
#   GITHUB_REPOSITORY       - Repository in owner/repo format
#   GITHUB_EVENT_NAME       - Event type (push, pull_request)
#   PR_HEAD_REPO_FULL_NAME  - For PRs: head repo in owner/repo format
#   IMAGE_TAG               - accel-sim-framework image tag

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

#==============================================================================
# CONFIGURATION
#==============================================================================

BUILD_JOBS=8
CONFIG="QV100-SASS"
BENCHMARKS="rodinia_2.0-ft"
TRACES_URL="https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz"
TRACE_PATH="./hw_run/rodinia_2.0-ft/9.1"
JOB_NAME="docker-short-$(ci_job_name)"

#==============================================================================
# STAGES
#==============================================================================

# On the runner, which has the SSH key for the gpgpu-sim forks
stage_setup() {
    clone_gpgpusim
    log_section "Getting the rodinia_2.0-ft traces"
    rm -rf ./hw_run/rodinia_2.0-ft
    mkdir -p ./hw_run
    wget -q "$TRACES_URL" -O rodinia_2.0-ft.tgz
    tar -xzf rodinia_2.0-ft.tgz -C ./hw_run
    rm rodinia_2.0-ft.tgz
}

stage_run() {
    log_section "Building and simulating in the ${IMAGE_TAG:-default} image"
    "$SCRIPT_DIR/on-cluster.sh" -c "$BUILD_JOBS" -t 04:00:00 -- \
        ./.github/scripts/main-docker.sh in-image
}

# Inside the image: its CUDA and compilers, no host environment
stage_in_image() {
    [ -n "$CUDA_INSTALL_PATH" ] || { log_error "CUDA_INSTALL_PATH not set: not in the CI image?"; exit 1; }
    [ -d ./gpu-simulator/gpgpu-sim ] || { log_error "no gpu-simulator/gpgpu-sim: run the setup stage first"; exit 1; }
    export PATH=$CUDA_INSTALL_PATH/bin:$PATH
    build_accelsim false "$BUILD_JOBS"
    source ./gpu-simulator/setup_environment.sh release
    run_simulations "$BENCHMARKS" "$CONFIG" "$TRACE_PATH" "$JOB_NAME"
    monitor_simulations "$JOB_NAME" 4 60 "stats-per-app-docker.csv" -v
}

#==============================================================================
# MAIN
#==============================================================================

case "${1:-all}" in
    setup) stage_setup ;;
    run) stage_run ;;
    in-image) stage_in_image ;;
    all) stage_setup; stage_run ;;
    *) echo "Unknown stage: $1"; exit 1 ;;
esac
