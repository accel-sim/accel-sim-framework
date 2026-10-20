#!/bin/bash
# Weekly CI: the steps of weekly.yml, one stage each.
#
#   ./weekly.sh <stage>
#
# Stages:
#   setup, build                       env-setup clone; gpgpu-sim clone + Accel-Sim build
#   trace-h200                         VPN, trace on the remote H200, publish (one Slurm job)
#   build-tracer-v100, trace-v100,     tracer and apps on the runner; traces and
#   hw-stats-v100, publish-v100        hardware stats on gpu01 (gpu-lock sruns); publish
#   simulate|correlate|archive <run>   run: h200, v100, or h200-ml (the hosted
#                                      ML traces of H200_ml_traces, never traced here)
#
# Requires GITHUB_REPOSITORY and BRANCH_NAME (set by weekly.yml); the trace-h200
# stage also VPN_HOST, VPN_CERT and REMOTE_HOST (org secrets).
# CI_SANDBOX=<dir> (lib/common.sh) redirects every side-effect path for local runs.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

H200_SUITES="rodinia_2.0-ft,rodinia-3.1,GPU_Microbenchmark,GPU_Microbenchmark_TMA,cutlass_tma_small"
V100_SUITES="rodinia_2.0-ft,rodinia-3.1,GPU_Microbenchmark"
ML_SUITES="llama_inference,llama_inference_2048,resnet_inference,bert_inference,dlrmv2_inference,unet3d_inference,retinanet_inference,sdxl_inference,mixtral_inference,deepseek_r1_inference,whisper_inference,qwen25_inference,qwen25_inference_2048,llama_training"
ML_DEPOT="$DEPOT/H200_ml_traces"
TRACE_DEVICE=7

export OPENCONNECT_PATH="/home/tgrogers-raid/a/common/openconnect/openconnect"
export OCPROXY_PATH="/home/tgrogers-raid/a/common/ocproxy/ocproxy"
export SOCKS_PORT="1080"
export REMOTE_USER="tgrghci"
REMOTE_WORK_DIR="/purdue/tgrghci/accel-sim-framework-ci${CI_SANDBOX:+-sandbox}"

#==============================================================================
# CONFIGURATION
#==============================================================================

require_env() {
    local v
    for v in "$@"; do
        if [ -z "${!v:-}" ]; then
            log_error "$v is not set"
            exit 1
        fi
    done
}

# This repo's trace depot of GPU $1 (each repo's sims read its own tracers' output)
depot_dir() {
    echo "$SHARED_ROOT/nightly-$1/${GITHUB_REPOSITORY//\//_}"
}

# Sets CONFIG, SUITES, TRACES, HW, MONITOR_HOURS and GROUP for run $1
gpu_vars() {
    MONITOR_HOURS=14 GROUP=ubench
    case "$1" in
        h200) CONFIG="H200-SASS"; SUITES="$H200_SUITES" ;;
        v100) CONFIG="QV100-SASS"; SUITES="$V100_SUITES" ;;
        h200-ml)
            # ~9K kernels (accelsim-layout.sh in the depot adds the traces/ and
            # device-0/ layout these paths read)
            CONFIG="H200-SASS" SUITES="$ML_SUITES" GROUP=ml MONITOR_HOURS=40
            TRACES="$ML_DEPOT/traces/device-0/12.8/" HW="$ML_DEPOT/"
            return
            ;;
        *)
            log_error "Run must be h200, v100 or h200-ml, not '${1:-}'"
            exit 1
            ;;
    esac
    HW="$(depot_dir "$1")/hw_run/"
    TRACES="${HW}traces/"
}

# Replace depot $2's hw_run with $1 (a finished trace + hardware-stat run)
publish_depot() {
    local new="$1" depot="$2" n
    n=$(find "$new/traces" -name kernelslist.g 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        log_error "No traces in $new: keeping $depot as it is"
        return 1
    fi
    mkdir -p "$depot"
    rm -rf "$depot/hw_run.old"
    if [ -e "$depot/hw_run" ]; then
        mv "$depot/hw_run" "$depot/hw_run.old"
    fi
    mv "$new" "$depot/hw_run"
    rm -rf "$depot/hw_run.old"
    chmod -R a+rX "$depot/hw_run"
    log_info "Published $n trace dirs to $depot/hw_run"
}

#==============================================================================
# STAGES
#==============================================================================

stage_setup() {
    setup_environment
}

stage_build() {
    source ./env-setup/12.8_env_setup.sh
    clone_gpgpusim
    build_accelsim
}

# Runs as one Slurm job (weekly.yml); the VPN goes down however it exits
stage_trace_h200() {
    require_env VPN_HOST VPN_CERT REMOTE_HOST
    local depot
    depot=$(depot_dir h200)
    trap vpn_cleanup EXIT
    vpn_connect
    remote_clean "$REMOTE_WORK_DIR"
    remote_sync push "$GITHUB_WORKSPACE" "$REMOTE_WORK_DIR"
    # Successful runs took 2h45m-3h47m (Aug-Sep 2026); a hang ends at 8h
    remote_exec "$REMOTE_WORK_DIR" "timeout --kill-after=5m 8h .github/scripts/h200.sh $H200_SUITES"
    rm -rf "$depot/hw_run.incoming"
    remote_sync pull "$REMOTE_WORK_DIR/hw_run" "$depot/hw_run.incoming"
    publish_depot "$depot/hw_run.incoming" "$depot"
}

stage_build_tracer_v100() {
    source ./env-setup/12.8_env_setup.sh
    log_section "Building the tracer"
    ./util/tracer_nvbit/install_nvbit.sh
    make clean -C ./util/tracer_nvbit/
    make -C ./util/tracer_nvbit/
    log_section "Building $V100_SUITES"
    export PATH=/home/tgrogers-raid/a/common/python2:$PATH
    rm -rf ./gpu-app-collection/
    git clone --quiet --recurse-submodules https://github.com/accel-sim/gpu-app-collection.git
    source ./gpu-app-collection/src/setup_environment
    ln -s /home/tgrogers-raid/a/common/data_dirs ./gpu-app-collection/
    make -j8 -C ./gpu-app-collection/src ${V100_SUITES//,/ }
}

# trace-v100 and hw-stats-v100 run on gpu01 inside the gpu-lock singleton
stage_trace_v100() {
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-app-collection/src/setup_environment
    rm -rf ./hw_run/
    "$SCRIPT_DIR/nvbit-trace.sh" -B "$V100_SUITES" -D "$TRACE_DEVICE"
}

stage_hw_stats_v100() {
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-app-collection/src/setup_environment
    ./util/hw_stats/run_hw.py -B "$V100_SUITES" -D "$TRACE_DEVICE"
}

stage_publish_v100() {
    publish_depot ./hw_run "$(depot_dir v100)"
}

stage_simulate() {
    gpu_vars "$1"
    local run="$1"
    set --  # sourced setup scripts would read $1 as their own argument
    source ./env-setup/12.8_env_setup.sh
    source ./gpu-simulator/setup_environment.sh
    local job_name="weekly-$run-$(ci_job_name)"
    run_simulations "$SUITES" "$CONFIG" "$TRACES" "$job_name" "${SIM_PER_KERNEL[@]}"
    # Weekly: one summary line per poll; a job the node's OOM killer took is not a failure
    monitor_simulations "$job_name" "$MONITOR_HOURS" 300 "weekly-$run-stats-per-app.csv" --allow-oom
    verify_simulations --per-kernel-dirs --allow-oom "$job_name" "$CONFIG:$SUITES"
}

stage_correlate() {
    gpu_vars "$1"
    set --
    source ./env-setup/12.8_env_setup.sh
    correlate_simulations "$CONFIG:$HW"
}

stage_archive() {
    gpu_vars "$1"
    local run="$1"
    set --
    source ./env-setup/12.8_env_setup.sh
    archive_runs "weekly/$GITHUB_REPOSITORY/$BRANCH_NAME" "Weekly $run" "$GROUP" "$CONFIG"
}

#==============================================================================
# MAIN DISPATCH
#==============================================================================

case "${1:-}" in
    -h|--help|help|"")
        sed -n '/^# Stages:/,/^# Requires/p' "$0" | sed '$d'
        exit 0
        ;;
esac
require_env GITHUB_REPOSITORY BRANCH_NAME

case "$1" in
    setup)             stage_setup ;;
    build)             stage_build ;;
    trace-h200)        stage_trace_h200 ;;
    build-tracer-v100) stage_build_tracer_v100 ;;
    trace-v100)        stage_trace_v100 ;;
    hw-stats-v100)     stage_hw_stats_v100 ;;
    publish-v100)      stage_publish_v100 ;;
    simulate)          stage_simulate "${2:-}" ;;
    correlate)         stage_correlate "${2:-}" ;;
    archive)           stage_archive "${2:-}" ;;
    *)
        log_error "Unknown stage: $1 (see $0 help)"
        exit 1
        ;;
esac
