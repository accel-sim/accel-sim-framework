#!/bin/bash
# Main CI SASS Simulation Script
#
# This script implements the SASS-Simulation job from main.yml.
# It can be run locally to reproduce the CI flow, or called with specific stages from CI.
#
# Usage:
#   ./main-sass.sh              # Run everything
#   ./main-sass.sh build        # Run specific stage
#   ./main-sass.sh simulate
#   ./main-sass.sh correlate
#   ./main-sass.sh archive
#
# Environment variables (set automatically in CI, or manually for local runs):
#   CI_TIER                 - smoke (default: PRs, branch pushes) or full (dev)
#   BRANCH_NAME             - Git branch name
#   GITHUB_REPOSITORY       - Repository in owner/repo format
#   GITHUB_EVENT_NAME       - Event type (push, pull_request)
#   GITHUB_RUN_NUMBER       - CI run number
#   GITHUB_RUN_ATTEMPT      - CI run attempt
#   CI_SANDBOX              - local runs: put every side-effect path under it

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

#==============================================================================
# CONFIGURATION
#==============================================================================

# Build settings
USE_SRUN=true
BUILD_JOBS=20

CI_TIER="${CI_TIER:-smoke}"

# Suites per tier and config ("<config>:<suites>"). smoke: fast suites, no
# archive or correlation; full: everything, archived and correlated.
case "$CI_TIER" in
    smoke)
        RUNS=(
            "QV100-SASS:GPU_Microbenchmark_CI,rodinia_2.0-ft"
            "H200-SASS:GPU_Microbenchmark_CI,rodinia_2.0-ft,GPU_Microbenchmark_TMA_CI,cutlass_tma_CI,hopper_features_CI"
        )
        ;;
    full)
        RUNS=(
            "QV100-SASS:GPU_Microbenchmark,rodinia_2.0-ft"
            "H200-SASS:GPU_Microbenchmark,rodinia_2.0-ft,cutlass_tma_small,GPU_Microbenchmark_TMA"
        )
        ;;
    *)
        echo "CI_TIER must be smoke or full, not '$CI_TIER'" >&2
        exit 1
        ;;
esac

# Traces and hardware stats of each config
trace_dir() {
    case "$1" in
        QV100-SASS) echo "$DEPOT/QV100/hw_run/traces/" ;;
        H200-SASS) echo "$DEPOT/H200/hw_run/traces/" ;;
        *) log_error "No trace dir for $1"; return 1 ;;
    esac
}
hw_dir() {
    case "$1" in
        QV100-SASS) echo "$DEPOT/QV100/hw_run/" ;;
        H200-SASS) echo "$DEPOT/H200/hw_run/" ;;
        *) log_error "No hardware dir for $1"; return 1 ;;
    esac
}
# The configs of RUNS, in order
run_configs() {
    local run
    for run in "${RUNS[@]}"; do echo "${run%%:*}"; done
}

# Statistics archive branch of this repo and branch
STATS_BRANCH="$GITHUB_REPOSITORY/$BRANCH_NAME"

# Job naming
JOB_NAME="sass-short-$(ci_job_name)"

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

stage_simulate() {
    log_section "Running SASS Simulations ($CI_TIER)"
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-simulator/setup_environment.sh

    rm -rf "$CI_STATS_DIR"
    local run config trace
    for run in "${RUNS[@]}"; do
        config="${run%%:*}"
        trace=$(trace_dir "$config")
        run_simulations "${run#*:}" "$config" "$trace" "$JOB_NAME"
    done

    monitor_simulations "$JOB_NAME" 12 300 "stats-per-app-sass.csv" -v
    verify_simulations "$JOB_NAME" "${RUNS[@]}"
}

stage_correlate() {
    source ./env-setup/12.8_env_setup.sh
    local config entries=()
    for config in $(run_configs); do
        entries+=("$config:$(hw_dir "$config")")
    done
    correlate_simulations "${entries[@]}"
}

stage_archive() {
    if [ "$CI_TIER" != full ] || [ "$GITHUB_EVENT_NAME" != push ]; then
        log_error "Only full-tier push runs are archived (tier $CI_TIER, event $GITHUB_EVENT_NAME)"
        return 1
    fi
    source ./env-setup/12.8_env_setup.sh
    # shellcheck disable=SC2046  # one word per config
    archive_runs "$STATS_BRANCH" "CI" ubench $(run_configs)

    local success_dir="$LAST_SUCCESS_ROOT/$GITHUB_REPOSITORY/$BRANCH_NAME"
    log_info "Saving to lastSuccess: $success_dir"
    rm -rf "$success_dir"
    mkdir -p "$success_dir"
    rsync -a "$PWD"/ "$success_dir/" || true
    chmod -R 777 "$success_dir"
}

# Run all stages
run_all() {
    stage_setup
    stage_build
    stage_simulate
    if [ "$CI_TIER" = full ]; then
        stage_correlate
    fi
    if [ "$CI_TIER" = full ] && [ "$GITHUB_EVENT_NAME" = push ]; then
        stage_archive
    fi
}

#==============================================================================
# MAIN DISPATCH
#==============================================================================

show_usage() {
    echo "Usage: $0 {setup|build|simulate|correlate|archive|all}"
    echo ""
    echo "Stages:"
    echo "  setup     - Clone and setup env-setup repository"
    echo "  build     - Clone gpgpu-sim and build accel-sim"
    echo "  simulate  - Run the CI_TIER suites and verify every simulation finished"
    echo "  correlate - Correlate every simulated config against hardware"
    echo "  archive   - Push this run's stats and correlation to statistics-archive"
    echo "  all       - Run all stages in order"
}

case "${1:-all}" in
    setup)     stage_setup ;;
    build)     stage_build ;;
    simulate)  stage_simulate ;;
    correlate) stage_correlate ;;
    archive)   stage_archive ;;
    all)       run_all ;;
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
