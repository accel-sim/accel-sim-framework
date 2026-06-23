#!/bin/bash
# Hopper Weekly CI Script
#
# This script implements both Tracer-Hopper and SASS-Hopper jobs from hopper-weekly.yml.
# It can be run locally to reproduce the full CI flow, or called with specific stages from CI.
#
# Usage:
#   ./hopper-weekly.sh              # Run everything (tracer + sass)
#   ./hopper-weekly.sh tracer       # Run all tracer stages
#   ./hopper-weekly.sh sass         # Run all sass stages
#   ./hopper-weekly.sh sass-build   # Run specific stage
#
# Available stages:
#   Tracer job:
#     tracer-vpn-connect, tracer-clean-remote, tracer-sync-push,
#     tracer-run, tracer-sync-pull, tracer-vpn-cleanup, tracer
#
#   SASS job:
#     sass-setup, sass-build, sass-simulate, sass-archive, sass-correlate, sass
#
# Environment variables (set automatically in CI, or manually for local runs):
#   BRANCH_NAME          - Git branch name
#   GITHUB_REPOSITORY    - Repository in owner/repo format
#   GITHUB_WORKSPACE     - Workspace directory (defaults to pwd)
#   GITHUB_RUN_NUMBER    - CI run number (for job naming)
#   GITHUB_RUN_ATTEMPT   - CI run attempt number

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

#==============================================================================
# CONFIGURATION
#==============================================================================

# VPN / Remote settings
export VPN_HOST="211.179.16.22"
export VPN_CERT="pin-sha256:t1qxDIv4DWnlmfO+m8Q1xKyPmnhuPtjEuf3KTa/3A28="
export OPENCONNECT_PATH="/home/tgrogers-raid/a/common/openconnect/openconnect"
export OCPROXY_PATH="/home/tgrogers-raid/a/common/ocproxy/ocproxy"
export SOCKS_PORT="1080"
export REMOTE_HOST="172.20.40.191"
export REMOTE_USER="tgrghci"
REMOTE_WORK_DIR="/purdue/tgrghci/accel-sim-framework-ci"

# Trace depot location
TRACE_DEPOT="/scratch/tgrogers-disk01/a/common/for-sharing/$USER/nightly-h200"

# SASS simulation settings
BENCHMARKS="rodinia_2.0-ft,rodinia-3.1,GPU_Microbenchmark"
BENCHMARKS_TMA="GPU_Microbenchmark_TMA,cutlass_tma_small"
BENCHMARKS_REGRESSION="rodinia_2.0-ft,rodinia-3.1,GPU_Microbenchmark"
CONFIG_H200="H200-SASS"
CONFIG_QV100="QV100-SASS"
CONFIG_A100="A100-SASS"
TRACE_PATH_H100="$TRACE_DEPOT/hw_run/traces/"
TRACE_PATH_QV100="/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/QV100/hw_run/traces/"
TRACE_PATH_A100="/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/A100/hw_run/traces/"
HW_RUN_H100="$TRACE_DEPOT/hw_run/"
HW_RUN_QV100="/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/QV100/hw_run/"
HW_RUN_A100="/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/A100/hw_run/"

# Statistics archive settings
STATS_BRANCH_PREFIX="hopper-weekly"
USE_SRUN=false

# Job naming
JOB_NAME="hopper-weekly-$(ci_job_name "")"

#==============================================================================
# TRACER STAGES
#==============================================================================

stage_tracer_vpn_connect() {
    vpn_connect
}

stage_tracer_clean_remote() {
    remote_clean "$REMOTE_WORK_DIR"
}

stage_tracer_sync_push() {
    log_info "Syncing repository to remote..."
    remote_sync push "$GITHUB_WORKSPACE" "$REMOTE_WORK_DIR"
}

stage_tracer_run() {
    log_section "Generating traces on H200"
    remote_exec "$REMOTE_WORK_DIR" ".github/scripts/h200.sh"
}

stage_tracer_sync_pull() {
    log_section "Syncing traces back from remote"
    mkdir -p ./hw_run
    remote_sync pull "$REMOTE_WORK_DIR/hw_run" "./hw_run"

    # Move to trace depot
    log_info "Moving traces to depot: $TRACE_DEPOT"
    rm -rf "$TRACE_DEPOT"
    mkdir -p "$TRACE_DEPOT"
    mv ./hw_run/ "$TRACE_DEPOT/"
    chmod 777 -R "$TRACE_DEPOT"
}

stage_tracer_vpn_cleanup() {
    vpn_cleanup
}

# Run all tracer stages
stage_tracer() {
    stage_tracer_vpn_connect
    stage_tracer_clean_remote
    stage_tracer_sync_push
    stage_tracer_run
    stage_tracer_sync_pull
    stage_tracer_vpn_cleanup
}

#==============================================================================
# SASS STAGES
#==============================================================================

stage_sass_setup() {
    setup_environment
}

stage_sass_build() {
    log_section "Building Accel-Sim"
    source ./env-setup/12.8_env_setup.sh
    clone_gpgpusim
    build_accelsim "$USE_SRUN"
}

stage_sass_simulate() {
    log_section "Running SASS Simulations"
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-simulator/setup_environment.sh

    # H200 simulations
    run_simulations "$BENCHMARKS" "$CONFIG_H200" "$TRACE_PATH_H100" "$JOB_NAME"
    run_simulations "$BENCHMARKS_TMA" "$CONFIG_H200" "$TRACE_PATH_H100" "$JOB_NAME"

    # QV100 regression
    run_simulations "$BENCHMARKS_REGRESSION" "$CONFIG_QV100" "$TRACE_PATH_QV100" "$JOB_NAME"

    # A100 regression
    run_simulations "$BENCHMARKS_REGRESSION" "$CONFIG_A100" "$TRACE_PATH_A100" "$JOB_NAME"

    # Monitor all jobs
    monitor_simulations "$JOB_NAME" 12 300 "hopper-weekly-stats-per-app.csv"
}

stage_sass_archive() {
    log_section "Archiving Statistics"
    source ./env-setup/12.8_env_setup.sh

    local stats_branch="$STATS_BRANCH_PREFIX/$BRANCH_NAME"
    setup_stats_archive "$stats_branch"

    # Collect and archive stats for every correlated GPU
    archive_config_stats "hopper-h100" "$BENCHMARKS,$BENCHMARKS_TMA" "$CONFIG_H200"
    archive_config_stats "ampere-a100" "$BENCHMARKS_REGRESSION" "$CONFIG_A100"
    archive_config_stats "v100" "$BENCHMARKS_REGRESSION" "$CONFIG_QV100"
}

stage_sass_correlate() {
    log_section "Running Correlation Analysis"
    source ./env-setup/12.8_env_setup.sh

    # Correlate each GPU independently; a failure for one must not drop the
    # others' plots or block the archive push.
    run_correlation "hopper-h100" "hopper-h100-ubench-sass-latest2.csv" "$HW_RUN_H100" \
        || log_error "hopper-h100 correlation failed"
    run_correlation "ampere-a100" "ampere-a100-ubench-sass-latest2.csv" "$HW_RUN_A100" \
        || log_error "ampere-a100 correlation failed"
    run_correlation "v100" "v100-ubench-sass-latest2.csv" "$HW_RUN_QV100" \
        || log_error "v100 correlation failed"

    push_stats_archive "Hopper Weekly" "$STATS_BRANCH_PREFIX/$BRANCH_NAME"
}

# Run all sass stages
stage_sass() {
    stage_sass_setup
    stage_sass_build
    stage_sass_simulate
    stage_sass_archive
    stage_sass_correlate
}

#==============================================================================
# MAIN DISPATCH
#==============================================================================

show_usage() {
    echo "Usage: $0 {tracer|sass|all|<stage-name>}"
    echo ""
    echo "Grouped stages:"
    echo "  tracer    - Run all tracer stages (VPN, sync, run, pull)"
    echo "  sass      - Run all SASS stages (setup, build, simulate, archive, correlate)"
    echo "  all       - Run tracer then sass"
    echo ""
    echo "Individual tracer stages:"
    echo "  tracer-vpn-connect, tracer-clean-remote, tracer-sync-push,"
    echo "  tracer-run, tracer-sync-pull, tracer-vpn-cleanup"
    echo ""
    echo "Individual sass stages:"
    echo "  sass-setup, sass-build, sass-simulate, sass-archive, sass-correlate"
}

case "${1:-all}" in
    # Individual tracer stages (for CI visibility)
    tracer-vpn-connect) stage_tracer_vpn_connect ;;
    tracer-clean-remote) stage_tracer_clean_remote ;;
    tracer-sync-push)   stage_tracer_sync_push ;;
    tracer-run)         stage_tracer_run ;;
    tracer-sync-pull)   stage_tracer_sync_pull ;;
    tracer-vpn-cleanup) stage_tracer_vpn_cleanup ;;
    tracer)             stage_tracer ;;

    # Individual sass stages (for CI visibility)
    sass-setup)     stage_sass_setup ;;
    sass-build)     stage_sass_build ;;
    sass-simulate)  stage_sass_simulate ;;
    sass-archive)   stage_sass_archive ;;
    sass-correlate) stage_sass_correlate ;;
    sass)           stage_sass ;;

    # Run everything
    all)
        stage_tracer
        stage_sass
        ;;

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
