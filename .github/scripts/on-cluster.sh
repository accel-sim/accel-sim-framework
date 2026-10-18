#!/bin/bash
# Run a command in the Accel-Sim CI image on a Slurm batch node.
#
#   .github/scripts/on-cluster.sh [-c CPUS] [-m MEM] [-t TIME] -- command [args...]
#
# For a job that used to run in a Docker container: the tgrogers-raid runner
# has srun but no container runtime, so the image runs under apptainer, as a
# SIF built once into CI_WORK and then only read. The command runs in
# $GITHUB_WORKSPACE (on /scratch, so any node can reach it) with /tmp in
# $GITHUB_WORKSPACE/tmp. The Slurm job carries the run in its comment, which
# cleanup-on-cancel uses to scancel it when the run is cancelled.
#
# Env: IMAGE_TAG (default ubuntu-24.04-cuda-12.8-minimal), PASS_ENV (names of
# variables to pass into the image), CI_WORK (where the SIFs live).
set -euo pipefail
cpus=8 mem=32G time=02:00:00
while getopts c:m:t: o; do
  case $o in c) cpus=$OPTARG ;; m) mem=$OPTARG ;; t) time=$OPTARG ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
[ "${1:-}" != -- ] || shift
[ $# -gt 0 ] || { echo "usage: $0 [-c CPUS] [-m MEM] [-t TIME] -- command..."; exit 2; }

IMAGE_TAG=${IMAGE_TAG:-ubuntu-24.04-cuda-12.8-minimal}
CI_WORK=${CI_WORK:-/scratch/tgrogers-disk01/a/tgrghci/ci/gpgpu-sim-regress}
SIF=$CI_WORK/$IMAGE_TAG.sif
# the gpgpu-sim CI keeps this image there already
[ "$IMAGE_TAG" != ubuntu-24.04-cuda-12.8-minimal ] || SIF=$CI_WORK/regress.sif
ws=${GITHUB_WORKSPACE:-$PWD}
tag=(--comment="gh-run:${GITHUB_RUN_ID:-local}.${GITHUB_RUN_ATTEMPT:-1}")
mkdir -p "$CI_WORK"

(
  flock 9
  if [ ! -f "$SIF" ]; then
    echo "building $SIF"
    # several GB of temporary space: not the node's small tmpfs
    srun -p batch -c 4 --mem=16G -t 03:00:00 -J ci-sif "${tag[@]}" bash -c "
      export APPTAINER_TMPDIR=\$(mktemp -d '$CI_WORK/sif-tmp.XXXX') APPTAINER_CACHEDIR=\$(mktemp -d '$CI_WORK/sif-cache.XXXX')
      trap 'rm -rf \$APPTAINER_TMPDIR \$APPTAINER_CACHEDIR' EXIT
      apptainer build -F '$SIF.new.$$' 'docker://ghcr.io/accel-sim/accel-sim-framework:$IMAGE_TAG' >/dev/null &&
        mv '$SIF.new.$$' '$SIF'"
  fi
) 9> "$SIF.lock"

rm -rf "$ws/tmp"; mkdir -p "$ws/tmp"
args=(--cleanenv --bind "$ws" --bind "$ws/tmp:/tmp" --env TMPDIR=/tmp
      --env "GITHUB_WORKSPACE=$ws"
      # git safe.directory without a writable /etc/gitconfig
      --env GIT_CONFIG_COUNT=1 --env GIT_CONFIG_KEY_0=safe.directory --env "GIT_CONFIG_VALUE_0=*")
for v in ${PASS_ENV:-}; do args+=(--env "$v=${!v}"); done
echo "srun: batch, $cpus cpus, $mem, $time, $IMAGE_TAG: $*"
exec srun -p batch -c "$cpus" --mem="$mem" -t "$time" -J "ci-${GITHUB_JOB:-job}" "${tag[@]}" \
  apptainer exec "${args[@]}" "$SIF" bash -c 'cd "$GITHUB_WORKSPACE" && exec "$@"' _ "$@"
