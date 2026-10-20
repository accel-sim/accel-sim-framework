#!/bin/bash
# Common functions for Accel-Sim CI scripts
#
# This library provides shared functions used across CI job scripts.
# Source this file at the beginning of each job script.
#
# Required environment variables (set by GitHub Actions or manually for local runs):
#   BRANCH_NAME          - Branch name to check out
#   GITHUB_REPOSITORY    - Repository in owner/repo format (e.g., accel-sim/accel-sim-framework)
#
# Optional environment variables:
#   GITHUB_EVENT_NAME       - Event type (push, pull_request, etc.)
#   PR_HEAD_REPO_FULL_NAME  - For PRs: head repo in owner/repo format
#   GITHUB_WORKSPACE        - Workspace directory (defaults to current directory)
#   GITHUB_RUN_NUMBER       - CI run number (for unique job names)
#   GITHUB_RUN_ATTEMPT      - CI run attempt number

set -eo pipefail

# Get script directory for relative path resolution
COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default GITHUB_WORKSPACE to current directory if not set
: "${GITHUB_WORKSPACE:=$(pwd)}"

#==============================================================================
# LOGGING
#==============================================================================

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_section() {
    echo ""
    echo "============================================"
    echo "$*"
    echo "============================================"
}

#==============================================================================
# SIDE-EFFECT PATHS
#==============================================================================

# Everything CI writes outside the workspace, plus its run outputs. CI leaves
# CI_SANDBOX unset; CI_SANDBOX=<dir> moves all of it under <dir> at once.
for _knob in TRACE_DEPOT TRACE_DEPOT_V100 REMOTE_WORK_DIR DEPOT STATS_BRANCH_OVERRIDE \
    STATS_ARCHIVE_URL STATS_ARCHIVE_DIR CI_STATS_DIR LAST_SUCCESS_ROOT; do
    if [ -n "${!_knob:-}" ]; then
        log_error "$_knob is not an override; CI_SANDBOX=<dir> redirects every side-effect path"
        exit 1
    fi
done
if [ -n "${CI_SANDBOX:-}" ]; then
    CI_SANDBOX=$(realpath -m "$CI_SANDBOX")
    mkdir -p "$CI_SANDBOX"
    # per-user dirs of for-sharing/ (the weekly trace depots)
    SHARED_ROOT="$CI_SANDBOX/for-sharing"
    STATS_ARCHIVE_URL="$CI_SANDBOX/statistics-archive.git"
    [ -d "$STATS_ARCHIVE_URL" ] || git init --quiet --bare "$STATS_ARCHIVE_URL"
    LAST_SUCCESS_ROOT="$CI_SANDBOX/lastSuccess"
    CI_OUT_DIR="$CI_SANDBOX"
else
    SHARED_ROOT="/scratch/tgrogers-disk01/a/common/for-sharing/$USER"
    STATS_ARCHIVE_URL="git@github.com:accel-sim/statistics-archive.git"
    LAST_SUCCESS_ROOT="/scratch/tgrogers-disk01/a/tgrghci/ci/lastSuccess"
    CI_OUT_DIR="$GITHUB_WORKSPACE"
fi
# Per-config results of this job: ci-stats/<GPU>-<config>/
CI_STATS_DIR="$CI_OUT_DIR/ci-stats"
STATS_ARCHIVE_DIR="$CI_OUT_DIR/statistics-archive"
# Read-only shared trace and hardware-stat depot
DEPOT="/scratch/tgrogers-disk01/a/common/for-sharing/accel-sim"

#==============================================================================
# ENVIRONMENT SETUP
#==============================================================================

# Clone and setup the env-setup repository
# Args: [branch] - defaults to "cluster-ubuntu"
setup_environment() {
    local branch="${1:-cluster-ubuntu}"
    log_section "Setting up environment (branch: $branch)"

    rm -rf env-setup
    git clone --quiet git@github.com:purdue-aalp/env-setup.git
    cd env-setup
    git checkout "$branch"
    cd ..

    log_info "Environment setup complete"
}

#==============================================================================
# GPGPU-SIM CLONE WITH FORK-AWARE BRANCH SELECTION
#==============================================================================

# Clone gpgpu-sim into ./gpu-simulator/gpgpu-sim: the same-named branch of the
# fork paired with the code's repo, else upstream accel-sim dev.
# Uses BRANCH_NAME, GITHUB_REPOSITORY, GITHUB_EVENT_NAME, PR_HEAD_REPO_FULL_NAME.
clone_gpgpusim() {
    local dest_dir="./gpu-simulator/gpgpu-sim"
    local upstream_url="git@github.com:accel-sim/gpgpu-sim_distribution.git"

    log_section "Cloning gpgpu-sim"
    : "${BRANCH_NAME:?BRANCH_NAME is not set}"
    : "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}"

    # The repo the code comes from: the PR's head repo (a fork such as
    # accel-sim-framework-public pairs with gpgpu-sim_distribution-public), else this repo
    local source_repo="$GITHUB_REPOSITORY"
    if [ "${GITHUB_EVENT_NAME:-}" = "pull_request" ] && [ -n "${PR_HEAD_REPO_FULL_NAME:-}" ]; then
        source_repo="$PR_HEAD_REPO_FULL_NAME"
    fi
    local owner="${source_repo%%/*}" repo="${source_repo#*/}"
    local fork="$owner/${repo/accel-sim-framework/gpgpu-sim_distribution}"
    local fork_url="git@github.com:$fork.git"

    rm -rf "$dest_dir"
    git clone --quiet "$upstream_url" "$dest_dir"

    # "Repository not found" (no such fork, e.g. a personal fork without one)
    # builds upstream dev; any other ls-remote failure is an error
    local out rc=0
    out=$(git ls-remote --heads "$fork_url" "refs/heads/$BRANCH_NAME" 2>&1) || rc=$?
    if [ "$rc" -ne 0 ]; then
        if [[ "$out" == *"Repository not found"* ]]; then
            log_info "No fork $fork: using upstream dev"
            git -C "$dest_dir" checkout --quiet -B dev origin/dev
        else
            echo "$out" >&2
            log_error "git ls-remote $fork_url failed (exit $rc)"
            return 1
        fi
    elif awk -v r="refs/heads/$BRANCH_NAME" '$2 == r {f = 1} END {exit !f}' <<< "$out"; then
        log_info "Using $fork branch $BRANCH_NAME"
        git -C "$dest_dir" fetch --quiet "$fork_url" \
            "refs/heads/$BRANCH_NAME:refs/remotes/fork/$BRANCH_NAME"
        git -C "$dest_dir" checkout --quiet -B "$BRANCH_NAME" "fork/$BRANCH_NAME"
    else
        log_info "No branch $BRANCH_NAME in $fork: using upstream dev"
        git -C "$dest_dir" checkout --quiet -B dev origin/dev
    fi
    log_info "gpgpu-sim at $(git -C "$dest_dir" rev-parse --abbrev-ref HEAD) $(git -C "$dest_dir" rev-parse --short HEAD)"
}

#==============================================================================
# BUILD
#==============================================================================

# Build accel-sim
# Args:
#   $1 - Use srun (true/false, default: false)
#   $2 - Number of parallel jobs (default: 20)
build_accelsim() {
    local use_srun="${1:-false}"
    local jobs="${2:-20}"

    log_section "Building Accel-Sim (srun: $use_srun, jobs: $jobs)"

    source ./gpu-simulator/setup_environment.sh release
    rm -rf ./gpu-simulator/build/release ./gpu-simulator/bin/release
    cmake -S ./gpu-simulator/ -B ./gpu-simulator/build/release

    if [ "$use_srun" = "true" ]; then
        srun --time=8:00:00 -c"$jobs" cmake --build ./gpu-simulator/build/release -j"$jobs"
    else
        cmake --build ./gpu-simulator/build/release -j"$jobs"
    fi
    cmake --install ./gpu-simulator/build/release

    log_info "Build complete"
}

#==============================================================================
# SIMULATIONS
#==============================================================================

# Run simulations with run_simulations.py
# Args:
#   $1 - Benchmarks (comma-separated)
#   $2 - Config
#   $3 - Trace path
#   $4 - Job name suffix (default: $$)
#   $5.. - Extra run_simulations.py flags (e.g. -M)
# Every CI run is per kernel: one Slurm array task per kernel (get_stats
# --per-kernel-dirs collects them), so a long kernel neither serialises the
# app's other kernels nor delays its own failure.
SIM_PER_KERNEL=(--per-kernel --array -M 10G)
run_simulations() {
    local benchmarks="$1"
    local config="$2"
    local trace_path="$3"
    local name_suffix="${4:-$$}"

    log_info "Running simulations: $benchmarks with config $config ${*:5}"
    ./util/job_launching/run_simulations.py \
        -B "$benchmarks" \
        -C "$config" \
        -T "$trace_path" \
        -N "$name_suffix" \
        "${@:5}"
}

# Monitor simulations with monitor_func_test.py
# Args:
#   $1 - Job name pattern
#   $2 - Timeout hours (default: 12)
#   $3 - Sleep time seconds (default: 300)
#   $4 - Output stats file (default: stats-per-app.csv)
#   $5.. - monitor_func_test.py flags (-v: every job's status each poll;
#          --allow-oom: a job killed for memory is not a failure)
monitor_simulations() {
    local name_pattern="$1"
    local timeout="${2:-12}"
    local sleep_time="${3:-300}"
    local stats_file="${4:-stats-per-app.csv}"

    log_info "Monitoring simulations: $name_pattern (timeout: ${timeout}h) ${*:5}"
    ./util/job_launching/monitor_func_test.py \
        -T "$timeout" \
        -S "$sleep_time" \
        -s "$stats_file" \
        -N "$name_pattern" \
        "${@:5}"
}

# Collect each "<config>:<suites>" run of launch $1 (run_simulations -N) into
# ci-stats/<config>/run.csv with get_stats --strict: a simulation without output
# or without GPGPU-Sim's terminating string fails the job instead of being
# dropped from the stats. Every run is collected before the failure is reported.
# Leading get_stats flags pass through (--per-kernel-dirs for --per-kernel runs;
# --allow-oom: a simulation killed for memory is only a warning).
verify_simulations() {
    log_section "Verifying simulation outputs"
    local flags=() entry config suites failed=()
    while [[ "${1:-}" == --* ]]; do
        flags+=("$1")
        shift
    done
    flags+=(-N "$1")
    shift
    for entry in "$@"; do
        config="${entry%%:*}" suites="${entry#*:}"
        mkdir -p "$CI_STATS_DIR/$config"
        log_info "Collecting $config: $suites"
        if ! ./util/job_launching/get_stats.py -k -K -R -A --strict "${flags[@]}" \
            -B "$suites" -C "$config" > "$CI_STATS_DIR/$config/run.csv"; then
            failed+=("$config")
        fi
    done
    if [ ${#failed[@]} -ne 0 ]; then
        log_error "Simulations without valid output in: ${failed[*]} (see ERROR --strict above)"
        return 1
    fi
    log_info "All simulations produced complete output"
}

#==============================================================================
# CORRELATION
#==============================================================================

# Correlate each "<config>:<hw_run dir>" against hardware (its stats from
# verify_simulations). Each config's plots, plot-correlation output and -S
# summary go to ci-stats/<config>/. Every config is correlated before a
# failure is reported.
correlate_simulations() {
    log_section "Running correlation analysis"
    local config hw entry out rc failed=()
    for entry in "$@"; do
        config="${entry%%:*}" hw="${entry#*:}"
        out="$CI_STATS_DIR/$config"
        if [ ! -s "$out/run.csv" ]; then
            log_error "No verified stats for $config ($out/run.csv)"
            failed+=("$config")
            continue
        fi
        log_info "Correlating $config with $hw"
        rm -rf ./util/plotting/correl-html/
        rc=0
        ./util/plotting/plot-correlation.py -c "$out/run.csv" -H "$hw" \
            -S "$out/summary.json" > "$out/correl.txt" || rc=$?
        awk '/^(NO-HW-REFERENCE|KERNEL-MISMATCH)/' "$out/correl.txt"
        if [ "$rc" -ne 0 ]; then
            failed+=("$config")
            continue
        fi
        cp ./util/plotting/correl-html/combined_per_kernel.html \
            ./util/plotting/correl-html/combined_per_app.html "$out/"
        find ./util/plotting/correl-html -name '*.kernel.raw.csv' -exec cp -t "$out/" {} +
    done
    if [ ${#failed[@]} -ne 0 ]; then
        log_error "Correlation failed for: ${failed[*]}"
        return 1
    fi
}

#==============================================================================
# STATISTICS ARCHIVE
#==============================================================================
#
# accel-sim/statistics-archive, one branch per repo and branch. Per group
# (ubench, or ml for the weekly's ML job) and config:
#   <group>/<GPU>-<config>/runs/<UTC time>_<run>-<attempt>_<sha>.csv   never rewritten
#   <group>/<GPU>-<config>/latest.csv, latest.txt                       the newest run
#   <group>/<GPU>-<config>/history.csv                    merge-stats of runs/, regenerated
#   <group>/<GPU>-<config>/{summary.json,correl.txt,combined_per_*.html}  newest correlation

# Clone statistics-archive and check out branch $1 as on the remote; a branch
# the remote does not have starts empty (never from another repo's stats).
setup_stats_archive() {
    local branch="$1"
    log_section "Setting up statistics-archive (branch: $branch)"
    if [ ! -d "$STATS_ARCHIVE_DIR/.git" ]; then
        git clone --quiet --no-checkout "$STATS_ARCHIVE_URL" "$STATS_ARCHIVE_DIR"
    fi
    stats_archive_reset "$branch"
}

stats_archive_reset() {
    local branch="$1"
    if git -C "$STATS_ARCHIVE_DIR" fetch --quiet origin "refs/heads/$branch:refs/remotes/origin/$branch" 2>/dev/null; then
        git -C "$STATS_ARCHIVE_DIR" checkout --quiet -B "$branch" "origin/$branch"
        git -C "$STATS_ARCHIVE_DIR" reset --quiet --hard "origin/$branch"
    else
        log_info "Branch $branch is new on the archive"
        git -C "$STATS_ARCHIVE_DIR" checkout --quiet --orphan "$branch"
        git -C "$STATS_ARCHIVE_DIR" rm -rf --quiet --ignore-unmatch .
    fi
    git -C "$STATS_ARCHIVE_DIR" clean -fdxq
}

# Put configs ${@:3}'s runs into the checked-out archive branch under group $2
# (re-runnable: the push retry calls it again on a fresh checkout)
stage_archive_files() {
    local run_id="$1" group="$2" config dir runs
    for config in "${@:3}"; do
        dir="$STATS_ARCHIVE_DIR/$group/$config"
        mkdir -p "$dir/runs"
        cp "$CI_STATS_DIR/$config/run.csv" "$dir/runs/$run_id.csv"
        cp "$CI_STATS_DIR/$config/run.csv" "$dir/latest.csv"
        echo "$run_id" > "$dir/latest.txt"
        cp "$CI_STATS_DIR/$config/"{summary.json,correl.txt,combined_per_kernel.html,combined_per_app.html} "$dir/"
        runs=$(find "$dir/runs" -maxdepth 1 -name '*.csv' | sort | paste -sd, -)
        ./util/plotting/merge-stats.py -R -c "$runs" > "$dir/history.csv"
    done
}

# Archive configs ${@:4} (verified and correlated) on statistics-archive branch
# $1 under group $3 (ubench, ml) and push. A rejected push (another job pushed
# first) refetches the branch and stages again; no -f.
archive_runs() {
    local branch="$1" msg="$2" group="$3"
    local configs=("${@:4}")
    local sha run_id attempt
    sha=$(git rev-parse --short=7 HEAD)
    run_id="$(date -u +%Y%m%d-%H%M%S)_${GITHUB_RUN_NUMBER:-local}-${GITHUB_RUN_ATTEMPT:-1}_$sha"
    setup_stats_archive "$branch"
    for attempt in 1 2 3 4 5; do
        stage_archive_files "$run_id" "$group" "${configs[@]}"
        git -C "$STATS_ARCHIVE_DIR" add --all
        if git -C "$STATS_ARCHIVE_DIR" diff --cached --quiet; then
            log_info "Nothing to archive"
            return 0
        fi
        git -C "$STATS_ARCHIVE_DIR" commit --quiet \
            -m "$msg CI automated checkin $branch Build:${GITHUB_RUN_NUMBER:-local}_${GITHUB_RUN_ATTEMPT:-1} ($run_id)"
        if git -C "$STATS_ARCHIVE_DIR" push --quiet origin "HEAD:refs/heads/$branch"; then
            log_info "Archived $run_id on $branch"
            return 0
        fi
        log_info "Push to $branch rejected (attempt $attempt), refetching"
        sleep $((attempt * 15))
        stats_archive_reset "$branch"
    done
    log_error "Could not push to statistics-archive $branch"
    return 1
}

#==============================================================================
# VPN AND REMOTE OPERATIONS
#==============================================================================

# Connect to VPN via openconnect with SOCKS proxy
# Required env vars: VPN_HOST, VPN_CERT, OPENCONNECT_PATH, OCPROXY_PATH, SOCKS_PORT
vpn_connect() {
    log_section "Connecting to VPN"

    # Validate required variables
    local required_vars="VPN_HOST VPN_CERT OPENCONNECT_PATH OCPROXY_PATH SOCKS_PORT"
    for var in $required_vars; do
        if [ -z "${!var}" ]; then
            log_error "Required variable $var not set"
            return 1
        fi
    done

    # A listener already there would pass the check below as ours. A killed
    # tracer's tunnel dies with its Slurm job (proctrack/cgroup), so it is not ours.
    if nc -z 127.0.0.1 "$SOCKS_PORT" 2>/dev/null; then
        log_error "SOCKS port $SOCKS_PORT is already in use"
        return 1
    fi

    log_info "Starting VPN connection to $VPN_HOST..."

    cat "$HOME/sk_password" | "$OPENCONNECT_PATH" \
        --pid-file="$(vpn_pid_file)" \
        --script-tun \
        --script "$OCPROXY_PATH -L 2222:unix-host:22 -D $SOCKS_PORT" \
        --protocol=gp \
        --user="$(cat "$HOME/sk_user")" \
        --servercert "$VPN_CERT" \
        --background \
        "$VPN_HOST"

    log_info "Waiting for VPN to establish..."
    sleep 5

    local i
    for i in {1..10}; do
        if nc -z 127.0.0.1 "$SOCKS_PORT" 2>/dev/null; then
            log_info "VPN connection established (SOCKS proxy ready on port $SOCKS_PORT)"
            return 0
        fi
        if [ "$i" -eq 10 ]; then
            log_error "VPN connection failed to establish"
            return 1
        fi
        sleep 2
    done
}

# Where vpn_connect leaves openconnect's PID for vpn_cleanup
vpn_pid_file() {
    echo "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/openconnect-${GITHUB_RUN_ID:-local}.pid"
}

# Cleanup the VPN connection this job started, and no other
vpn_cleanup() {
    log_section "Cleaning up VPN connection"
    local pid_file pid
    pid_file=$(vpn_pid_file)
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
        log_info "VPN cleanup complete (openconnect pid $pid)"
    else
        log_info "No VPN started by this job ($pid_file missing)"
    fi
}

# A stalled tunnel ends the session after 5 minutes instead of at the job limit
REMOTE_KEEPALIVE="-o ServerAliveInterval=60 -o ServerAliveCountMax=5"

# ssh REMOTE_USER@REMOTE_HOST through the VPN's SOCKS proxy
remote_ssh() {
    # shellcheck disable=SC2086  # REMOTE_KEEPALIVE is a list of options
    ssh -o "ProxyCommand=nc -X 5 -x 127.0.0.1:$SOCKS_PORT %h %p" $REMOTE_KEEPALIVE \
        "$REMOTE_USER@$REMOTE_HOST" "$@"
}

# Sync files to/from remote via SOCKS proxy
# Args:
#   $1 - Mode: "push" or "pull"
#   $2 - Source path
#   $3 - Destination path
# Required env vars: SOCKS_PORT, REMOTE_USER, REMOTE_HOST
remote_sync() {
    local mode="$1"
    local src="$2"
    local dest="$3"

    log_section "Remote sync: $mode"

    local ssh_opts="ssh -o 'ProxyCommand=nc -X 5 -x 127.0.0.1:$SOCKS_PORT %h %p' $REMOTE_KEEPALIVE"

    # The transfer rides a userspace SOCKS5 proxy over the VPN, which drops the
    # connection on long single transfers (broken pipe / socket IO errors).
    # Keep it resilient: exclude the repo's .git history (the remote only builds
    # the tracer and never runs git on the synced tree), keep partial files so a
    # dropped connection resumes, time out stalled sockets, and retry.
    local rsync_opts=(-az --partial --timeout=120 --exclude='.git')
    local from to
    case "$mode" in
        push)
            log_info "Syncing $src to $REMOTE_USER@$REMOTE_HOST:$dest"
            from="$src/"
            to="$REMOTE_USER@$REMOTE_HOST:$dest/"
            ;;
        pull)
            log_info "Syncing $REMOTE_USER@$REMOTE_HOST:$src to $dest"
            mkdir -p "$dest"
            from="$REMOTE_USER@$REMOTE_HOST:$src/"
            to="$dest/"
            ;;
        *)
            log_error "Unknown mode: $mode (expected 'push' or 'pull')"
            return 1
            ;;
    esac

    local attempt max_attempts=5
    for attempt in $(seq 1 "$max_attempts"); do
        if rsync "${rsync_opts[@]}" -e "$ssh_opts" "$from" "$to"; then
            log_info "Sync complete"
            return 0
        fi
        log_error "rsync attempt $attempt/$max_attempts failed; retrying after backoff..."
        sleep $((attempt * 10))
    done

    log_error "rsync failed after $max_attempts attempts"
    return 1
}

# Execute command on remote host via SOCKS proxy
# Args:
#   $1 - Remote working directory
#   $2 - Command to execute
# Required env vars: SOCKS_PORT, REMOTE_USER, REMOTE_HOST
remote_exec() {
    local work_dir="$1"
    local cmd="$2"

    log_section "Remote execution"
    log_info "Host: $REMOTE_USER@$REMOTE_HOST"
    log_info "Dir:  $work_dir"
    log_info "Cmd:  $cmd"

    remote_ssh bash -s <<ENDSSH
set -e
cd "$work_dir"
$cmd
ENDSSH

    log_info "Remote execution complete"
}

# Clean remote directory $1 before sync. A cancelled tracer's remote processes
# outlive its ssh session; only one tracer uses the dir (the singleton srun), so
# every process of this user working inside it is a leftover: kill, then remove.
# Required env vars: SOCKS_PORT, REMOTE_USER, REMOTE_HOST
remote_clean() {
    local remote_path="$1"

    log_info "Cleaning remote directory: $remote_path"

    remote_ssh bash -s <<'ENDSSH' -- "$remote_path"
set -e
dir=$1
leftovers() {
    local p
    for p in $(pgrep -u "$(id -un)"); do
        [ "$p" = $$ ] && continue
        case "$(readlink "/proc/$p/cwd" 2>/dev/null)" in
            "$dir"|"$dir"/*) echo "$p" ;;
        esac
    done
}
pids=$(leftovers)
if [ -n "$pids" ]; then
    echo "Killing leftover processes in $dir: $(echo $pids)"
    kill $pids 2>/dev/null || true
    sleep 10
    pids=$(leftovers)
    [ -z "$pids" ] || kill -9 $pids 2>/dev/null || true
    sleep 2
fi
if [ -d "$dir" ]; then
    echo "Removing existing directory..."
    rm -rf "$dir"
fi
ENDSSH
}

#==============================================================================
# UTILITY
#==============================================================================

# Unique suffix of this CI run's job names
ci_job_name() {
    echo "${GITHUB_RUN_NUMBER:-$$}_${GITHUB_RUN_ATTEMPT:-1}"
}
