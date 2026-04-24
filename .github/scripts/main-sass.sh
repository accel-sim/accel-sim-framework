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
#   ./main-sass.sh archive
#   ./main-sass.sh correlate
#
# Environment variables (set automatically in CI, or manually for local runs):
#   BRANCH_NAME             - Git branch name
#   GITHUB_REPOSITORY       - Repository in owner/repo format
#   GITHUB_EVENT_NAME       - Event type (push, pull_request)
#   PR_HEAD_REPO_FULL_NAME  - For PRs: head repo in owner/repo format
#   GITHUB_RUN_NUMBER       - CI run number
#   GITHUB_RUN_ATTEMPT      - CI run attempt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

#==============================================================================
# CONFIGURATION
#==============================================================================

# Build settings
USE_SRUN=true
BUILD_JOBS=20

# GPU configs and trace paths
CONFIGS=(
    "QV100-SASS:/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/QV100/hw_run/traces/"
    "A100-SASS:/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/A100/hw_run/traces/"
    "H200-SASS:/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/H200/hw_run/traces/"
)

# HW paths for correlation (config:hw_path)
HW_PATHS=(
    "QV100-SASS:/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/QV100/hw_run/"
    "A100-SASS:/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/A100/hw_run/"
    "H200-SASS:/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim/H200/hw_run/"
)

# Benchmarks per config
BENCHMARKS_STANDARD="GPU_Microbenchmark,rodinia_2.0-ft"
BENCHMARKS_H200_EXTRA="cutlass_tma_small,GPU_Microbenchmark_TMA"

# Stat file prefixes (config:prefix)
STAT_PREFIXES=(
    "QV100-SASS:v100"
    "A100-SASS:ampere-a100"
    "H200-SASS:hopper-h200"
)

# Job naming
JOB_NAME="sass-short-$(ci_job_name "")"

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
    log_section "Running SASS Simulations"
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-simulator/setup_environment.sh

    # Run simulations for each config
    for config_path in "${CONFIGS[@]}"; do
        local config="${config_path%%:*}"
        local trace_path="${config_path#*:}"

        run_simulations "$BENCHMARKS_STANDARD" "$config" "$trace_path" "$JOB_NAME"

        # H200 has extra benchmarks
        if [[ "$config" == "H200-SASS" ]]; then
            run_simulations "$BENCHMARKS_H200_EXTRA" "$config" "$trace_path" "$JOB_NAME"
        fi
    done

    # Monitor all jobs
    monitor_simulations "$JOB_NAME" 12 300 "stats-per-app-sass.csv"
}

stage_archive() {
    log_section "Archiving Statistics"
    source ./env-setup/12.8_env_setup.sh

    # Determine branch name for stats archive
    local stats_branch
    if [ -n "$PR_HEAD_REPO_FULL_NAME" ]; then
        stats_branch="$PR_HEAD_REPO_FULL_NAME/$BRANCH_NAME"
    else
        stats_branch="$GITHUB_REPOSITORY/$BRANCH_NAME"
    fi

    setup_stats_archive "$stats_branch"

    # Collect and merge stats for each config
    for prefix_config in "${STAT_PREFIXES[@]}"; do
        local config="${prefix_config%%:*}"
        local prefix="${prefix_config#*:}"

        log_info "Collecting stats for $config ($prefix)"

        # Get stats
        ./util/job_launching/get_stats.py -k -K -R -B GPU_Microbenchmark -C "$config" -A \
            | tee "${prefix}-ubench-sass-local.csv"

        # Merge with archive (cumulative)
        ./util/plotting/merge-stats.py -R \
            -c "./statistics-archive/ubench/${prefix}-ubench-sass.csv,${prefix}-ubench-sass-local.csv" \
            | tee "${prefix}-ubench-sass.csv" \
            && mv "${prefix}-ubench-sass.csv" ./statistics-archive/ubench/

        # Merge latest (for correlation) and save as new latest
        ./util/plotting/merge-stats.py -R \
            -c "./statistics-archive/ubench/${prefix}-ubench-sass-latest.csv,${prefix}-ubench-sass-local.csv" \
            | tee "${prefix}-ubench-sass-latest2.csv"

        mv "${prefix}-ubench-sass-local.csv" "./statistics-archive/ubench/${prefix}-ubench-sass-latest.csv"
    done
}

stage_correlate() {
    log_section "Running Correlation Analysis"
    source ./env-setup/12.8_env_setup.sh

    ./util/hw_stats/get_hw_data.sh > /dev/null 2>&1 || true

    # Run correlation for each config
    for prefix_config in "${STAT_PREFIXES[@]}"; do
        local config="${prefix_config%%:*}"
        local prefix="${prefix_config#*:}"

        # Find matching HW path
        local hw_path=""
        for hw_entry in "${HW_PATHS[@]}"; do
            if [[ "$hw_entry" == "$config:"* ]]; then
                hw_path="${hw_entry#*:}"
                break
            fi
        done

        if [ -z "$hw_path" ]; then
            log_error "No HW path found for $config"
            continue
        fi

        log_info "Correlating $prefix with HW data from $hw_path"

        rm -rf ./util/plotting/correl-html/
        ./util/plotting/plot-correlation.py \
            -c "./${prefix}-ubench-sass-latest2.csv" \
            -H "$hw_path" \
            | tee "${prefix}-ubench-correl.txt"

        mv ./util/plotting/correl-html/combined_per_kernel.html \
            "./statistics-archive/ubench/${prefix}-combined_per_kernel.html"
        mv ./util/plotting/correl-html/combined_per_app.html \
            "./statistics-archive/ubench/${prefix}-combined_per_app.html"
    done

    # Commit and push
    local stats_branch
    if [ -n "$PR_HEAD_REPO_FULL_NAME" ]; then
        stats_branch="$PR_HEAD_REPO_FULL_NAME/$BRANCH_NAME"
    else
        stats_branch="$GITHUB_REPOSITORY/$BRANCH_NAME"
    fi

    push_stats_archive "CI" "$stats_branch"

    # On push events, save to lastSuccess directory
    if [ "$GITHUB_EVENT_NAME" = "push" ]; then
        local success_dir="/scratch/tgrogers-disk01/a/tgrghci/ci/lastSuccess/$GITHUB_REPOSITORY/$BRANCH_NAME"
        log_info "Saving to lastSuccess: $success_dir"
        rm -rf "$success_dir"
        mkdir -p "$success_dir"
        rsync -a "$PWD"/ "$success_dir/" || true
        chmod -R 777 "$success_dir"
    fi
}

# Run all stages
run_all() {
    stage_setup
    stage_build
    stage_simulate
    stage_archive
    stage_correlate
}

#==============================================================================
# MAIN DISPATCH
#==============================================================================

show_usage() {
    echo "Usage: $0 {setup|build|simulate|archive|correlate|all}"
    echo ""
    echo "Stages:"
    echo "  setup     - Clone and setup env-setup repository"
    echo "  build     - Clone gpgpu-sim and build accel-sim"
    echo "  simulate  - Run SASS simulations for QV100, A100, H200"
    echo "  archive   - Collect stats and merge with statistics-archive"
    echo "  correlate - Run correlation analysis and push results"
    echo "  all       - Run all stages in order"
}

case "${1:-all}" in
    setup)     stage_setup ;;
    build)     stage_build ;;
    simulate)  stage_simulate ;;
    archive)   stage_archive ;;
    correlate) stage_correlate ;;
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
