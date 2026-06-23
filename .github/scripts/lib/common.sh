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

set -e

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

# Clone gpgpu-sim with fork-aware branch selection
# This allows coordinated changes across accel-sim-framework and gpgpu-sim repositories.
#
# Args:
#   $1 - Destination directory (default: ./gpu-simulator/gpgpu-sim)
#   $2 - Fallback branch (default: dev)
#
# Uses environment variables:
#   BRANCH_NAME, GITHUB_EVENT_NAME, PR_HEAD_REPO_FULL_NAME, GITHUB_REPOSITORY
clone_gpgpusim() {
    local dest_dir="${1:-./gpu-simulator/gpgpu-sim}"
    local fallback_branch="${2:-dev}"
    local upstream_url="git@github.com:accel-sim/gpgpu-sim_distribution.git"

    log_section "Cloning gpgpu-sim with fork-aware branch selection"

    # Validate required variables
    if [ -z "$BRANCH_NAME" ]; then
        log_error "BRANCH_NAME not set"
        return 1
    fi

    # Determine fork owner
    local fork_owner
    if [ "$GITHUB_EVENT_NAME" = "pull_request" ] && [ -n "$PR_HEAD_REPO_FULL_NAME" ]; then
        fork_owner=$(echo "$PR_HEAD_REPO_FULL_NAME" | cut -d'/' -f1)
        log_info "Detected PR event, using owner from PR head: $fork_owner"
    elif [ -n "$GITHUB_REPOSITORY" ]; then
        fork_owner=$(echo "$GITHUB_REPOSITORY" | cut -d'/' -f1)
        log_info "Using owner from GITHUB_REPOSITORY: $fork_owner"
    else
        log_error "Cannot determine fork owner. Set GITHUB_REPOSITORY."
        return 1
    fi

    # Derive gpgpu-sim repo name from accel-sim-framework
    local framework_repo gpgpusim_repo
    if [ -n "$GITHUB_REPOSITORY" ]; then
        framework_repo=$(echo "$GITHUB_REPOSITORY" | cut -d'/' -f2)
        gpgpusim_repo=$(echo "$framework_repo" | sed 's/accel-sim-framework/gpgpu-sim_distribution/')
    else
        gpgpusim_repo="gpgpu-sim_distribution"
    fi

    local fork_url="git@github.com:$fork_owner/$gpgpusim_repo.git"

    log_info "  Destination:     $dest_dir"
    log_info "  Branch:          $BRANCH_NAME"
    log_info "  Fork owner:      $fork_owner"
    log_info "  Fork repo:       $gpgpusim_repo"
    log_info "  Fallback branch: $fallback_branch"

    # Remove existing directory
    rm -rf "$dest_dir"

    # Clone upstream repository
    log_info "Cloning upstream gpgpu-sim..."
    if ! git clone --quiet "$upstream_url" "$dest_dir" 2>&1; then
        log_error "Failed to clone upstream repository"
        return 2
    fi

    # Try to add fork as remote and checkout matching branch
    log_info "Attempting to checkout branch '$BRANCH_NAME' from '$fork_owner/$gpgpusim_repo'"

    if git -C "$dest_dir" remote add fork-owner "$fork_url" 2>/dev/null; then
        if git -C "$dest_dir" fetch fork-owner 2>/dev/null; then
            if git -C "$dest_dir" rev-parse --verify "fork-owner/$BRANCH_NAME" >/dev/null 2>&1; then
                log_info "Found branch '$BRANCH_NAME' in '$fork_owner/$gpgpusim_repo', checking it out"
                git -C "$dest_dir" checkout -B "$BRANCH_NAME" "fork-owner/$BRANCH_NAME"
            else
                # Branch not found in fork, try fork's default branch
                local default_branch
                default_branch=$(git -C "$dest_dir" remote show fork-owner 2>/dev/null | grep 'HEAD branch' | awk '{print $NF}')
                if [ -n "$default_branch" ]; then
                    log_info "Branch '$BRANCH_NAME' not found, using default branch '$default_branch' from '$fork_owner/$gpgpusim_repo'"
                    git -C "$dest_dir" checkout -B "$default_branch" "fork-owner/$default_branch"
                else
                    log_info "Could not determine fork's default branch, falling back to upstream '$fallback_branch'"
                    git -C "$dest_dir" checkout -B "$fallback_branch" "origin/$fallback_branch"
                fi
            fi
        else
            log_info "Could not fetch from fork, falling back to upstream '$fallback_branch'"
            git -C "$dest_dir" checkout -B "$fallback_branch" "origin/$fallback_branch"
        fi
        # Clean up temporary remote
        git -C "$dest_dir" remote remove fork-owner 2>/dev/null || true
    else
        log_info "Could not add '$fork_owner/$gpgpusim_repo' remote, falling back to upstream '$fallback_branch'"
        git -C "$dest_dir" checkout -B "$fallback_branch" "origin/$fallback_branch"
    fi

    # Report final state
    local final_branch final_commit
    final_branch=$(git -C "$dest_dir" rev-parse --abbrev-ref HEAD)
    final_commit=$(git -C "$dest_dir" rev-parse --short HEAD)
    log_info "Successfully cloned gpgpu-sim"
    log_info "  Branch: $final_branch"
    log_info "  Commit: $final_commit"
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
run_simulations() {
    local benchmarks="$1"
    local config="$2"
    local trace_path="$3"
    local name_suffix="${4:-$$}"

    log_info "Running simulations: $benchmarks with config $config"
    ./util/job_launching/run_simulations.py \
        -B "$benchmarks" \
        -C "$config" \
        -T "$trace_path" \
        -N "$name_suffix"
}

# Monitor simulations with monitor_func_test.py
# Args:
#   $1 - Job name pattern
#   $2 - Timeout hours (default: 12)
#   $3 - Sleep time seconds (default: 300)
#   $4 - Output stats file (default: stats-per-app.csv)
monitor_simulations() {
    local name_pattern="$1"
    local timeout="${2:-12}"
    local sleep_time="${3:-300}"
    local stats_file="${4:-stats-per-app.csv}"

    log_info "Monitoring simulations: $name_pattern (timeout: ${timeout}h)"
    ./util/job_launching/monitor_func_test.py \
        -T "$timeout" \
        -S "$sleep_time" \
        -v \
        -s "$stats_file" \
        -N "$name_pattern"
}

#==============================================================================
# STATISTICS ARCHIVE
#==============================================================================

# Clone and setup statistics-archive repository
# Args:
#   $1 - Branch name (will be created if doesn't exist)
setup_stats_archive() {
    local branch="$1"

    log_section "Setting up statistics-archive (branch: $branch)"

    rm -rf ./statistics-archive
    git clone --quiet git@github.com:accel-sim/statistics-archive.git

    # Checkout or create branch
    git -C ./statistics-archive checkout "$branch" 2>/dev/null || \
        git -C ./statistics-archive checkout -b "$branch"

    mkdir -p statistics-archive/ubench/
    log_info "Statistics archive ready"
}

# Merge stats CSV files
# Args:
#   $1 - Archive CSV path (in statistics-archive/ubench/)
#   $2 - Local CSV path
merge_stats() {
    local archive_csv="$1"
    local local_csv="$2"
    local output_csv
    output_csv=$(basename "$archive_csv")

    ./util/plotting/merge-stats.py -R -c "./statistics-archive/ubench/$archive_csv,$local_csv" \
        | tee "$output_csv" && mv "$output_csv" "./statistics-archive/ubench/"
}

# Collect sim stats for one GPU, fold them into the archive's cumulative history,
# and emit the merged-with-latest CSV that correlation consumes.
# Args:
#   $1 - GPU prefix (e.g. "hopper-h100", "ampere-a100", "v100")
#   $2 - Benchmark list
#   $3 - Config name (e.g. "H200-SASS")
# Produces: ./statistics-archive/ubench/${prefix}-ubench-sass.csv (history),
#           ./statistics-archive/ubench/${prefix}-ubench-sass-latest.csv (this run),
#           ./${prefix}-ubench-sass-latest2.csv (latest + this run, for correlation)
archive_config_stats() {
    local prefix="$1"
    local benchmarks="$2"
    local config="$3"
    local local_csv="${prefix}-ubench-sass-local.csv"
    local latest_csv="${prefix}-ubench-sass-latest.csv"

    log_section "Collecting stats for $prefix ($config)"

    ./util/job_launching/get_stats.py -k -K -R \
        -B "$benchmarks" -C "$config" -A \
        | tee "$local_csv"

    # Fold into cumulative history
    merge_stats "${prefix}-ubench-sass.csv" "$local_csv"

    # Merge the previous latest with this run for correlation
    ./util/plotting/merge-stats.py -R \
        -c "./statistics-archive/ubench/${latest_csv},${local_csv}" \
        | tee "${prefix}-ubench-sass-latest2.csv"

    # Save this run's stats as the new latest
    mv "$local_csv" "./statistics-archive/ubench/${latest_csv}"
}

# Commit and push statistics-archive
# Args:
#   $1 - Commit message prefix
#   $2 - Branch name
push_stats_archive() {
    local msg_prefix="$1"
    local branch="$2"
    local run_info="${GITHUB_RUN_NUMBER:-local}_${GITHUB_RUN_ATTEMPT:-1}"

    log_section "Pushing statistics archive"

    git -C ./statistics-archive add --all
    git -C ./statistics-archive commit \
        -m "$msg_prefix CI automated checkin $branch Build:$run_info" || echo "No changes to commit"
    git -C ./statistics-archive push -f -u origin "$branch"

    log_info "Statistics archive pushed"
}

#==============================================================================
# CORRELATION
#==============================================================================

# Run correlation analysis
# Args:
#   $1 - GPU name prefix (e.g., "hopper-h100", "v100")
#   $2 - Stats CSV file
#   $3 - HW run path
run_correlation() {
    local gpu_prefix="$1"
    local stats_csv="$2"
    local hw_run_path="$3"

    log_section "Running correlation for $gpu_prefix"

    ./util/hw_stats/get_hw_data.sh > /dev/null 2>&1 || true
    rm -rf ./util/plotting/correl-html/

    ./util/plotting/plot-correlation.py -c "$stats_csv" -H "$hw_run_path" \
        | tee "${gpu_prefix}-ubench-correl.txt"

    mv ./util/plotting/correl-html/combined_per_kernel.html \
        "./statistics-archive/ubench/${gpu_prefix}-combined_per_kernel.html"
    mv ./util/plotting/correl-html/combined_per_app.html \
        "./statistics-archive/ubench/${gpu_prefix}-combined_per_app.html"

    log_info "Correlation complete"
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

    log_info "Starting VPN connection to $VPN_HOST..."

    cat "$HOME/sk_password" | "$OPENCONNECT_PATH" \
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

# Cleanup VPN connection
# Required env vars: VPN_HOST
vpn_cleanup() {
    log_section "Cleaning up VPN connection"
    pkill -f "openconnect.*$VPN_HOST" || true
    log_info "VPN cleanup complete"
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

    local ssh_opts="ssh -o 'ProxyCommand=nc -X 5 -x 127.0.0.1:$SOCKS_PORT %h %p'"

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

    ssh -o "ProxyCommand=nc -X 5 -x 127.0.0.1:$SOCKS_PORT %h %p" \
        "$REMOTE_USER@$REMOTE_HOST" bash -s <<ENDSSH
set -e
cd "$work_dir"
$cmd
ENDSSH

    log_info "Remote execution complete"
}

# Clean remote directory before sync
# Args:
#   $1 - Remote directory path
# Required env vars: SOCKS_PORT, REMOTE_USER, REMOTE_HOST
remote_clean() {
    local remote_path="$1"

    log_info "Cleaning remote directory: $remote_path"

    ssh -o "ProxyCommand=nc -X 5 -x 127.0.0.1:$SOCKS_PORT %h %p" \
        "$REMOTE_USER@$REMOTE_HOST" bash -s <<ENDSSH
if [ -d "$remote_path" ]; then
    echo "Removing existing directory..."
    rm -rf "$remote_path"
fi
ENDSSH
}

#==============================================================================
# UTILITY
#==============================================================================

# Generate unique job name for this CI run
# Args:
#   $1 - Prefix
ci_job_name() {
    local prefix="$1"
    local run_id="${GITHUB_RUN_NUMBER:-$$}_${GITHUB_RUN_ATTEMPT:-1}"
    echo "${prefix}-${run_id}"
}
