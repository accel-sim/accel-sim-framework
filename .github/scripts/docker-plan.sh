#!/bin/bash
# Decide which dockerfiles/<tag>.Dockerfile images to build. Prints a JSON array of tags on stdout,
# and one "build/skip <tag>: <reason>" line per image on stderr (and in the job summary).
#   workflow_dispatch: the tags in $INPUT_IMAGES (space separated).
#   pull_request / push: Dockerfiles changed between $BASE and $HEAD. A modified file always
#   builds; an added one builds only if its tag is not on ghcr yet, so importing or renaming
#   existing images does not rebuild them.
# Env: EVENT, BASE, HEAD, INPUT_IMAGES, IMAGE (ghcr repository, e.g. accel-sim/accel-sim-framework)
set -euo pipefail

on_ghcr() {
  local token code
  token=$(curl -fsS "https://ghcr.io/token?scope=repository:$IMAGE:pull&service=ghcr.io" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')
  code=$(curl -s -o /dev/null -w '%{http_code}' -I -H "Authorization: Bearer $token" \
    -H 'Accept: application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json' \
    "https://ghcr.io/v2/$IMAGE/manifests/$1")
  case $code in
    200) return 0 ;;
    404) return 1 ;;
    *) echo "ghcr manifest check for $1 returned HTTP $code (expected 200 or 404)" >&2; exit 1 ;;
  esac
}

tags=()
say() { echo "$*" >&2; [ -z "${GITHUB_STEP_SUMMARY:-}" ] || echo "- $*" >> "$GITHUB_STEP_SUMMARY"; }
if [ "$EVENT" = workflow_dispatch ]; then
  for t in $INPUT_IMAGES; do
    [ -f "dockerfiles/$t.Dockerfile" ] || { echo "no dockerfiles/$t.Dockerfile" >&2; exit 1; }
    tags+=("$t"); say "build $t: requested"
  done
else
  [[ $BASE =~ ^0+$ ]] && BASE=$HEAD~1  # first push of a branch has no "before"
  while read -r status path; do
    t=$(basename "$path" .Dockerfile)
    case $status in
      M) tags+=("$t"); say "build $t: modified" ;;
      A) if on_ghcr "$t"; then say "skip $t: added, already on ghcr"; else tags+=("$t"); say "build $t: new image"; fi ;;
      D) say "skip $t: deleted" ;;
      *) echo "unexpected git status $status for $path" >&2; exit 1 ;;
    esac
  done < <(git diff --no-renames --name-status "$BASE" "$HEAD" -- 'dockerfiles/*.Dockerfile')
fi
[ ${#tags[@]} -gt 0 ] || say "nothing to build"

printf '%s\n' "${tags[@]}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))'
