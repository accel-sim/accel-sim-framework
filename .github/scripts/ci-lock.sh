#!/bin/bash
# A lock shared by the CI of every repo on this cluster, for scheduled runs
# that use the same host resources (remote dir, VPN, SOCKS port, trace depot,
# stats branch). GitHub's concurrency groups are per repo, so the accel-sim
# and purdue-aalp runs of one workflow would otherwise overlap.
#
#   ci-lock.sh acquire NAME [MAX_WAIT_MIN]   wait for NAME (default 600 min)
#   ci-lock.sh release NAME
#
# The lock is a directory (mkdir is atomic, also on NFS) holding its owner:
# repo, run id, attempt, start time. It belongs to a run, not a job: acquire
# again from a later job of the same run succeeds, and the run releases it
# once at the end. A lock whose run is no longer queued or in progress
# (GitHub API) is stale and broken; when the API cannot tell, a lock older
# than CI_LOCK_STALE_MIN is.
#
# Env: GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GH_TOKEN
# (optional), CI_LOCK_ROOT, CI_LOCK_STALE_MIN (default 1500), CI_LOCK_POLL_SEC
# (default 60).
set -euo pipefail

cmd=${1:?usage: $0 acquire|release NAME [MAX_WAIT_MIN]}
name=${2:?usage: $0 acquire|release NAME [MAX_WAIT_MIN]}
max_wait_min=${3:-600}
root=${CI_LOCK_ROOT:-/scratch/tgrogers-disk01/a/tgrghci/ci/locks}
stale_min=${CI_LOCK_STALE_MIN:-1500}
poll=${CI_LOCK_POLL_SEC:-60}
lock=$root/$name
repo=${GITHUB_REPOSITORY:-local}
run=${GITHUB_RUN_ID:-local-$$}
mkdir -p "$root"

field() { sed -n "s/^$1=//p" "$lock/owner" 2>/dev/null || true; }

# 0: the owner run is queued or in progress, 1: it is over, 2: unknown
owner_active() {
  local orepo orun status
  orepo=$(field repo) orun=$(field run)
  [ -n "$orepo" ] && [ -n "$orun" ] || return 2
  case $orun in local*) return 2 ;; esac
  status=$(curl -sSf -m 20 ${GH_TOKEN:+-H "Authorization: Bearer $GH_TOKEN"} \
    "https://api.github.com/repos/$orepo/actions/runs/$orun" | jq -r '.status // empty') || return 2
  case $status in
    completed) return 1 ;;
    "") return 2 ;;
    *) return 0 ;;
  esac
}

lock_age_min() {
  local t
  t=$(field started)
  [ -n "$t" ] || t=$(stat -c %Y "$lock" 2>/dev/null || date +%s)
  echo $(( ($(date +%s) - t) / 60 ))
}

take() {
  mkdir "$lock" 2>/dev/null || return 1
  printf 'repo=%s\nrun=%s\nattempt=%s\nstarted=%s\nhost=%s\n' "$repo" "$run" \
    "${GITHUB_RUN_ATTEMPT:-1}" "$(date +%s)" "$(hostname)" > "$lock/owner.tmp"
  mv "$lock/owner.tmp" "$lock/owner"
}

mine() { [ "$(field repo)" = "$repo" ] && [ "$(field run)" = "$run" ]; }

# Break the lock if stale; serialized so two waiters cannot both break it and
# one remove the lock the other just took
break_if_stale() {
  (
    flock 9
    [ -d "$lock" ] || exit 0
    # an owner file is written right after mkdir; none for a while is a crash
    if [ ! -f "$lock/owner" ]; then
      [ "$(lock_age_min)" -ge 5 ] || exit 0
      why="no owner file"
    else
      rc=0; owner_active || rc=$?
      age=$(lock_age_min)
      if [ $rc -eq 1 ]; then why="run $(field repo)#$(field run) is over"
      elif [ $rc -eq 2 ] && [ "$age" -ge "$stale_min" ]; then why="held ${age} min, run status unknown"
      else exit 0
      fi
    fi
    echo "breaking stale lock $name ($why)"
    rm -rf "$lock"
  ) 9> "$lock.break"
}

case $cmd in
  acquire)
    deadline=$(( $(date +%s) + max_wait_min * 60 ))
    last=0
    while :; do
      if take; then echo "acquired lock $name"; exit 0; fi
      if mine; then echo "lock $name already held by this run"; exit 0; fi
      break_if_stale
      take && { echo "acquired lock $name"; exit 0; }
      now=$(date +%s)
      if [ "$now" -ge "$deadline" ]; then
        echo "::error::lock $name still held by $(field repo)#$(field run) after ${max_wait_min} min"
        exit 1
      fi
      if [ $((now - last)) -ge 600 ]; then
        echo "waiting for lock $name, held by $(field repo)#$(field run) for $(lock_age_min) min"
        last=$now
      fi
      sleep "$poll"
    done
    ;;
  release)
    if [ ! -d "$lock" ]; then echo "lock $name is not held"
    elif mine; then rm -rf "$lock"; echo "released lock $name"
    else echo "::warning::lock $name is held by $(field repo)#$(field run), not this run: left alone"
    fi
    ;;
  *) echo "usage: $0 acquire|release NAME [MAX_WAIT_MIN]"; exit 2 ;;
esac
