#!/bin/bash
# Copy what explains a failed CI simulation run into DEST, for upload as an
# artifact (the next job on the runner wipes the workspace): the job launcher's
# logfiles (sim log, the monitor's node_details verdicts, failed_job_log) and
# each run dir's .o/.e, gpgpusim.config and justrun.sh. Run dirs hold a traces
# symlink into the trace depot, so find does not follow links.
#
#   .github/scripts/collect-sim-logs.sh DEST [ACCELSIM_ROOT]
set -u
dest=$(mkdir -p "${1:?usage: collect-sim-logs.sh DEST [ACCELSIM_ROOT]}" && cd "$1" && pwd)
root=$(cd "${2:-$(dirname "$0")/../..}" && pwd)
cd "$root" || exit 0
[ -d util/job_launching/logfiles ] && cp -r util/job_launching/logfiles "$dest/"
for d in sim_run_*; do
  [ -d "$d" ] || continue
  find -P "$d" -type f \( -regex '.*\.[oe][0-9]+' -o -name gpgpusim.config -o -name justrun.sh \) -print0 |
    xargs -0 -r cp --parents -t "$dest"
done
echo "collect-sim-logs: $(find "$dest" -type f | wc -l) files, $(du -sh "$dest" | cut -f1) in $dest"
