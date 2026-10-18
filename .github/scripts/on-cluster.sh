#!/bin/bash
# Run a CI command on a Slurm compute node inside the CI image, from the
# tgrogers-raid runner (which has srun but no container runtime). The
# workspace is on /scratch, so whichever node srun picks can read it.
#
#   .github/scripts/on-cluster.sh [-c CPUS] [-m MEM] [-t TIME] -- command [args...]
#
# IMAGE_TAG picks the accel-sim-framework image (what the job's container used
# to be); the apps are the image's own built gpu-app-collection (GPUAPPS_ROOT
# from the image). The command runs in $GITHUB_WORKSPACE with /tmp =
# $GITHUB_WORKSPACE/tmp, emptied each run. Everything it writes is in the
# job's own workspace; the image is only read. Variables named in PASS_ENV are
# passed in. When
# GitHub cancels the run, its cleanup-on-cancel job scancels every Slurm job
# tagged with the run (--comment): killing srun alone leaves the job running.
set -euo pipefail
cpus=8 mem=32G time=02:00:00
while getopts c:m:t: o; do
  case $o in c) cpus=$OPTARG ;; m) mem=$OPTARG ;; t) time=$OPTARG ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
[ "${1:-}" != -- ] || shift
[ $# -gt 0 ] || { echo "usage: $0 [-c CPUS] [-m MEM] [-t TIME] -- command..."; exit 2; }

IMAGE_TAG=${IMAGE_TAG:-ubuntu-24.04-cuda-12.8-minimal}
IMAGE=docker://ghcr.io/accel-sim/accel-sim-framework:$IMAGE_TAG
CI_WORK=${CI_WORK:-/scratch/tgrogers-disk01/a/tgrghci/ci/gpgpu-sim-regress}
SIF=$CI_WORK/$IMAGE_TAG.sif
# the funtional repo's CI already keeps this image there
[ "$IMAGE_TAG" != ubuntu-24.04-cuda-12.8-minimal ] || SIF=$CI_WORK/regress.sif
ws=${GITHUB_WORKSPACE:-$PWD}
mkdir -p "$CI_WORK"
tag=(--comment="gh-run:${GITHUB_RUN_ID:-local}.${GITHUB_RUN_ATTEMPT:-1}")

# The image as a SIF, one per tag and the only shared file: built once, then only read. A new
# build goes to a temporary name and replaces it atomically, under a lock.
(
  flock 9
  if [ ! -f "$SIF" ]; then
    echo "building $SIF from $IMAGE"
    # the build needs several GB of temporary space: not the node's 3 GB tmpfs
    srun -p batch -c 4 --mem=16G -t 03:00:00 -J ci-sif "${tag[@]}" bash -c "
      export APPTAINER_TMPDIR=\$(mktemp -d '$CI_WORK/sif-tmp.XXXX') APPTAINER_CACHEDIR=\$(mktemp -d '$CI_WORK/sif-cache.XXXX')
      trap 'rm -rf \$APPTAINER_TMPDIR \$APPTAINER_CACHEDIR' EXIT
      apptainer build -F '$SIF.new.$$' '$IMAGE' >/dev/null && mv '$SIF.new.$$' '$SIF'"
  fi
) 9> "$SIF.lock"

# The job's /tmp (compiler temporaries, the job launcher's staging area,
# mktemp) in its workspace: private to the job, and not the node's small tmpfs.
rm -rf "$ws/tmp"; mkdir -p "$ws/tmp"
args=(--cleanenv --bind "$ws" --bind "$ws/tmp:/tmp"
      --env TMPDIR=/tmp
      --env "GITHUB_WORKSPACE=$ws"
      # git safe.directory without writing a shared ~/.gitconfig
      --env GIT_CONFIG_COUNT=1 --env GIT_CONFIG_KEY_0=safe.directory --env "GIT_CONFIG_VALUE_0=*")
for v in ${PASS_ENV:-}; do args+=(--env "$v=${!v}"); done
echo "srun: batch, $cpus cpus, $mem, $time: $*"
exec srun -p batch -c "$cpus" --mem="$mem" -t "$time" -J "ci-${GITHUB_JOB:-job}" "${tag[@]}" \
  apptainer exec "${args[@]}" "$SIF" bash -c 'cd "$GITHUB_WORKSPACE" && exec "$@"' _ "$@"
