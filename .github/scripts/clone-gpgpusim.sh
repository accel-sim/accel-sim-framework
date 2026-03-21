#!/bin/bash
# Clone gpgpu-sim with fork-aware branch selection
#
# This script clones gpgpu-sim and attempts to check out a matching branch
# from the same fork owner. This allows coordinated changes across both
# accel-sim-framework and gpgpu-sim repositories.
#
# Usage:
#   ./clone-gpgpusim.sh [options]
#
# Options:
#   -d, --dest <path>       Destination directory (default: ./gpu-simulator/gpgpu-sim)
#   -b, --branch <name>     Branch name to look for (default: $BRANCH_NAME env var)
#   -o, --owner <name>      Fork owner to try (default: auto-detect from env)
#   -f, --fallback <branch> Fallback branch if nothing found (default: dev)
#   -q, --quiet             Suppress informational output
#   -h, --help              Show this help message
#
# Environment variables (auto-detected if not specified via options):
#   BRANCH_NAME             - Branch name to check out (required)
#   GITHUB_EVENT_NAME       - GitHub event type (pull_request, push, etc.)
#   PR_HEAD_REPO_FULL_NAME  - Full name of PR head repo (owner/repo)
#   GITHUB_REPOSITORY       - Current repository (owner/repo)
#
# Exit codes:
#   0 - Success
#   1 - Missing required parameters
#   2 - Clone failed
#   3 - Checkout failed
#
# Examples:
#   # Standard CI usage (uses env vars)
#   ./clone-gpgpusim.sh
#
#   # Explicit parameters for local testing
#   ./clone-gpgpusim.sh -b my-feature -o myuser -d ./gpu-simulator/gpgpu-sim
#
#   # With custom fallback branch
#   ./clone-gpgpusim.sh --fallback main

set -e

# Defaults
DEST_DIR="./gpu-simulator/gpgpu-sim"
UPSTREAM_URL="git@github.com:accel-sim/gpgpu-sim_distribution.git"
FALLBACK_BRANCH="dev"
QUIET=false
FORK_OWNER=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dest)
            DEST_DIR="$2"
            shift 2
            ;;
        -b|--branch)
            BRANCH_NAME="$2"
            shift 2
            ;;
        -o|--owner)
            FORK_OWNER="$2"
            shift 2
            ;;
        -f|--fallback)
            FALLBACK_BRANCH="$2"
            shift 2
            ;;
        -u|--upstream)
            UPSTREAM_URL="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            head -40 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper function for logging
log() {
    if [ "$QUIET" = false ]; then
        echo "$@"
    fi
}

# Validate required parameters
if [ -z "$BRANCH_NAME" ]; then
    echo "ERROR: BRANCH_NAME not set. Either set the environment variable or use -b/--branch."
    exit 1
fi

# Auto-detect fork owner if not specified
if [ -z "$FORK_OWNER" ]; then
    if [ "$GITHUB_EVENT_NAME" = "pull_request" ] && [ -n "$PR_HEAD_REPO_FULL_NAME" ]; then
        FORK_OWNER=$(echo "$PR_HEAD_REPO_FULL_NAME" | cut -d'/' -f1)
        log "Detected PR event, using owner from PR head: $FORK_OWNER"
    elif [ -n "$GITHUB_REPOSITORY" ]; then
        FORK_OWNER=$(echo "$GITHUB_REPOSITORY" | cut -d'/' -f1)
        log "Using owner from GITHUB_REPOSITORY: $FORK_OWNER"
    else
        echo "ERROR: Cannot determine fork owner. Set GITHUB_REPOSITORY or use -o/--owner."
        exit 1
    fi
fi

# Derive gpgpu-sim repo name from accel-sim-framework
if [ -n "$GITHUB_REPOSITORY" ]; then
    FRAMEWORK_REPO=$(echo "$GITHUB_REPOSITORY" | cut -d'/' -f2)
    GPGPUSIM_REPO=$(echo "$FRAMEWORK_REPO" | sed 's/accel-sim-framework/gpgpu-sim_distribution/')
else
    GPGPUSIM_REPO="gpgpu-sim_distribution"
fi

FORK_URL="git@github.com:$FORK_OWNER/$GPGPUSIM_REPO.git"

log "============================================"
log "Clone gpgpu-sim with fork-aware branch selection"
log "============================================"
log "  Destination:     $DEST_DIR"
log "  Branch:          $BRANCH_NAME"
log "  Fork owner:      $FORK_OWNER"
log "  Fork repo:       $GPGPUSIM_REPO"
log "  Upstream URL:    $UPSTREAM_URL"
log "  Fallback branch: $FALLBACK_BRANCH"
log ""

# Remove existing directory
rm -rf "$DEST_DIR"

# Clone upstream repository
log "Cloning upstream gpgpu-sim..."
if ! git clone --quiet "$UPSTREAM_URL" "$DEST_DIR" 2>&1; then
    echo "ERROR: Failed to clone upstream repository"
    exit 2
fi

# Try to add fork as remote and checkout matching branch
log "Attempting to checkout branch '$BRANCH_NAME' from '$FORK_OWNER/$GPGPUSIM_REPO'"

if git -C "$DEST_DIR" remote add fork-owner "$FORK_URL" 2>/dev/null; then
    # Successfully added remote, try to fetch
    if git -C "$DEST_DIR" fetch fork-owner 2>/dev/null; then
        # Check if the branch exists in the fork
        if git -C "$DEST_DIR" rev-parse --verify "fork-owner/$BRANCH_NAME" >/dev/null 2>&1; then
            log "Found branch '$BRANCH_NAME' in '$FORK_OWNER/$GPGPUSIM_REPO', checking it out"
            git -C "$DEST_DIR" checkout -B "$BRANCH_NAME" "fork-owner/$BRANCH_NAME"
        else
            # Branch not found in fork, try fork's default branch
            default_branch=$(git -C "$DEST_DIR" remote show fork-owner 2>/dev/null | grep 'HEAD branch' | awk '{print $NF}')
            if [ -n "$default_branch" ]; then
                log "Branch '$BRANCH_NAME' not found, using default branch '$default_branch' from '$FORK_OWNER/$GPGPUSIM_REPO'"
                git -C "$DEST_DIR" checkout -B "$default_branch" "fork-owner/$default_branch"
            else
                log "Could not determine fork's default branch, falling back to upstream '$FALLBACK_BRANCH'"
                git -C "$DEST_DIR" checkout -B "$FALLBACK_BRANCH" "origin/$FALLBACK_BRANCH"
            fi
        fi
    else
        log "Could not fetch from fork, falling back to upstream '$FALLBACK_BRANCH'"
        git -C "$DEST_DIR" checkout -B "$FALLBACK_BRANCH" "origin/$FALLBACK_BRANCH"
    fi
    # Clean up temporary remote
    git -C "$DEST_DIR" remote remove fork-owner 2>/dev/null || true
else
    log "Could not add '$FORK_OWNER/$GPGPUSIM_REPO' remote, falling back to upstream '$FALLBACK_BRANCH'"
    git -C "$DEST_DIR" checkout -B "$FALLBACK_BRANCH" "origin/$FALLBACK_BRANCH"
fi

# Report final state
FINAL_BRANCH=$(git -C "$DEST_DIR" rev-parse --abbrev-ref HEAD)
FINAL_COMMIT=$(git -C "$DEST_DIR" rev-parse --short HEAD)
log ""
log "Successfully cloned gpgpu-sim"
log "  Branch: $FINAL_BRANCH"
log "  Commit: $FINAL_COMMIT"

exit 0
