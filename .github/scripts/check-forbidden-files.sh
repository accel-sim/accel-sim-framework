#!/bin/bash
# Detect forbidden files added in any commit (even if later removed)
# This catches the case where someone commits large files then removes them,
# but the blob remains in git history bloating the repo.
#
# Usage:
#   ./check-forbidden-files.sh                    # Uses GitHub env vars or defaults
#   ./check-forbidden-files.sh <base> <head>      # Local testing with explicit refs
#
# Environment variables (for GitHub Actions):
#   CHECK_BASE_REF  - Explicit base branch (e.g., "dev")
#   GITHUB_BASE_REF - Auto-set by GitHub for PRs
#   GITHUB_SHA      - Auto-set by GitHub for current commit
#
# Examples:
#   ./check-forbidden-files.sh origin/dev HEAD
#   ./check-forbidden-files.sh upstream/dev aalp-private/Multigpu-Accelsim
#   CHECK_BASE_REF=dev ./check-forbidden-files.sh

set -e

# Determine base and head refs
# Priority: CLI args > CHECK_BASE_REF env var > GITHUB_BASE_REF (PR) > default
if [ $# -eq 2 ]; then
    BASE_REF="$1"
    HEAD_REF="$2"
elif [ -n "$CHECK_BASE_REF" ]; then
    # Explicit base ref passed from workflow
    BASE_REF="origin/$CHECK_BASE_REF"
    HEAD_REF="${GITHUB_SHA:-HEAD}"
elif [ -n "$GITHUB_BASE_REF" ]; then
    # In GitHub Actions PR context - GITHUB_BASE_REF is the target branch name
    BASE_REF="origin/$GITHUB_BASE_REF"
    HEAD_REF="${GITHUB_SHA:-HEAD}"
else
    # Local testing or push context - default to origin/dev
    BASE_REF="origin/dev"
    HEAD_REF="${GITHUB_SHA:-HEAD}"
fi

echo "Checking for forbidden files in commits: $BASE_REF..$HEAD_REF"

# Forbidden patterns - files that should never be committed
# These match .gitignore entries and trace file patterns
FORBIDDEN_PATTERNS=(
    # Directories that should not be committed
    '^hw_run/'
    '^env-setup/'
    '^gpu-app-collection/'
    '^sim_run_'
    '^4\.2/'
    '^extern/'
    '^debug_config/'
    '/traces/'

    # Trace and simulation output files
    '\.traceg$'
    '\.trace$'
    '\.mem$'
    '\.sass$'

    # Binary/compiled files
    '\.so$'
    '\.a$'
    '\.dylib$'

    # Large data/output files
    '\.pdf$'
    '\.ppm$'
    '\.bin$'
    '\.tgz$'
    '\.tar\.gz$'

    # Job output files (Slurm/Torque)
    '\.o[0-9]+$'
    '\.e[0-9]+$'

    # Misc
    '\.DS_Store'
    'gpucomputingsdk.*\.run$'
    '\.csv$'
)

# Build grep pattern
GREP_PATTERN=$(IFS='|'; echo "${FORBIDDEN_PATTERNS[*]}")

# Get all files added in any commit in the range (catches add-then-remove)
FORBIDDEN_FILES=$(git log --diff-filter=A --name-only --pretty=format:"" "$BASE_REF".."$HEAD_REF" 2>/dev/null | \
    grep -E "$GREP_PATTERN" | \
    sort -u || true)

# Check for large files (e.g., notebooks with embedded outputs)
# Size limit in bytes (512KB)
SIZE_LIMIT=524288
SIZE_LIMIT_HUMAN="512KB"

LARGE_FILES=""
while IFS= read -r file; do
    [ -z "$file" ] && continue
    # Get the blob size for this file at HEAD_REF
    size=$(git cat-file -s "$HEAD_REF:$file" 2>/dev/null || echo 0)
    if [ "$size" -gt "$SIZE_LIMIT" ]; then
        size_mb=$(echo "scale=1; $size / 1048576" | bc)
        LARGE_FILES="${LARGE_FILES}${file} (${size_mb}MB)\n"
    fi
done < <(git log --diff-filter=A --name-only --pretty=format:"" "$BASE_REF".."$HEAD_REF" 2>/dev/null | \
    grep -E '\.ipynb$' | sort -u || true)

HAS_ERRORS=0

if [ -n "$FORBIDDEN_FILES" ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Forbidden files detected in commits!"
    echo "=========================================="
    echo ""
    echo "The following files were added in one or more commits."
    echo "Even if removed later, they bloat the git history."
    echo ""
    echo "$FORBIDDEN_FILES" | head -50

    TOTAL=$(echo "$FORBIDDEN_FILES" | wc -l)
    if [ "$TOTAL" -gt 50 ]; then
        echo ""
        echo "... and $((TOTAL - 50)) more files"
    fi

    echo ""
    echo "Please remove these files from git history using:"
    echo "  git rebase -i $BASE_REF"
    echo "  # Then edit the commit(s) that added these files"
    HAS_ERRORS=1
fi

if [ -n "$LARGE_FILES" ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Large notebook files detected!"
    echo "=========================================="
    echo ""
    echo "The following .ipynb files exceed ${SIZE_LIMIT_HUMAN}:"
    echo ""
    echo -e "$LARGE_FILES"
    echo "Notebooks with embedded outputs (especially Plotly charts) can be huge."
    echo "Please clear outputs before committing:"
    echo "  jupyter nbconvert --clear-output --inplace <notebook.ipynb>"
    HAS_ERRORS=1
fi

if [ "$HAS_ERRORS" -eq 1 ]; then
    echo ""
    exit 1
fi

echo "No forbidden files found."
exit 0
