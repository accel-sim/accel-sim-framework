# CI statistics, correlation and archive

## Flow

| Trigger | Suites | Correlate | Archive |
|---|---|---|---|
| pull_request (smoke) | QV100/H200: GPU_Microbenchmark_CI, rodinia_2.0-ft; H200: GPU_Microbenchmark_TMA_CI, cutlass_tma_CI, hopper_features_CI | no | no |
| push to dev (full) | QV100/H200: GPU_Microbenchmark, rodinia_2.0-ft; H200: cutlass_tma_small, GPU_Microbenchmark_TMA | yes | yes |
| weekly | H200 (this week's traces): rodinia, ubench, TMA, cutlass_tma_small; QV100 (this week's V100 traces): rodinia, ubench; H200 ML (static H200 depot, 14 suites, 20 runs, ~9K kernels) | all | all |

Each SASS job: simulate (`run_simulations`), monitor, `verify_simulations`
(`get_stats --strict` per config into `ci-stats/<config>/run.csv`),
`correlate_simulations` (`plot-correlation -S`), `archive_runs`, each given the
configs of the script's run table. Every step needs the earlier ones to succeed.
Refuted: a run manifest (`ci-run-manifest.tsv`) the stages iterated and a
`get_stats --manifest` JSON archived per run; the run tables say the same and
`--strict` names any missing sim.

## Why (audit findings, Oct 2026)

- Failed runs were archived (`!cancelled()`/`always()`): commit ea642c0 on
  `hopper-weekly/dev` dropped 4216 lines and left 0-byte files.
- `merge-stats.py` printed only the last file's apps (the `union_*` sets were
  intersections and unused), so an app missing from one run vanished from the
  cumulative CSV. On `hopper-weekly/dev` v100 history + latest, the old merge
  also lost 98 values of the latest run that the new one keeps.
- `get_stats` only warned on sims without a terminating string: run 131 of dev
  passed with 6 H200 sims dropped.
- Correlation errors were swallowed by `|| log_error`; `correl-html/` was not
  created when no figure was made (FileNotFoundError).
- `set("NVIDIA H200 80GB HBM3")` is a set of characters, so any device with an
  "N" matched. Real names: `NVIDIA H200`, `NVIDIA A100 80GB PCIe`,
  `Tesla V100-SXM2-32GB`, `NVIDIA H100 80GB HBM3` (H100-newapps, h100_ml_traces).
- `accel-sim/H100` in the depot is a link to `H200` (its stats say
  `NVIDIA H200`), so those traces run as H200-SASS.
- Kernels were paired by position. ncu names are demangled with a different
  format than c++filt (`const float *` vs `float const*`), so pairing compares
  the qualified name without template and parameter lists. On QV100 and H200
  ubench + rodinia_2.0-ft (210 kernels each) both pairings agree; dropping one
  backprop kernel from the sim CSV mis-paired the other positionally, while
  name pairing still matches the right hardware kernel.

## Archive

`accel-sim/statistics-archive`, branch `<repo>/<branch>` for main.yml dev
pushes and `weekly/<repo>/<branch>` for weekly, for every repo alike. Per
config dir `<group>/<config>/` (group `ubench`, or `ml`): `runs/` (one CSV per run),
`latest.csv`, `latest.txt`, `history.csv` (merge of `runs/`), and the newest
`summary.json`, `correl.txt`, `combined_per_*.html`. Nothing is imported from
the pre-redesign files; the old `hopper-weekly/*` branches stay as history.

Refuted: rebasing on a rejected push. `latest.csv` and `history.csv` conflict
whenever two jobs push; refetching and restaging is conflict-free because each
job only rewrites its own config dirs.

## No run-to-run comparison

CI keeps the pre-redesign behaviour: correlate with hardware and archive the
CSVs. Refuted: `compare-stats.py` (per-kernel diff against the previous dev
run, first with per-stat tolerances, then without); the archived `runs/`
CSVs can be diffed by hand when needed.

## Weekly

- Merges dev's `hopper-weekly.yml` and Volta `weekly.yml`, written from
  scratch (PR #146's port was dropped). Jobs as in `main.yml`: Tracer-H200 ->
  SASS-H200 and Tracer-V100 -> SASS-V100, independent of each other; one
  `weekly.sh` stage per step. No A100, no static-depot regression runs.
- `concurrency: weekly-<repo>`: the depots are per repo, not per run, so a
  second run's tracer would replace the traces the first run's sims read.
- SASS-H200-ML replays the ML traces in the static depot
  `accel-sim/H200/hw_run` (`traces/device-7/12.8/<w>/<a>/traces`, HW in
  `device-7/12.8/<w>/<a>/{ncu.csv.nsight,ncu.csv.gpc__cycles_elapsed.0}`),
  copied 2026-10-20 from `accel-sim/H200_ml_traces` (another user's depot:
  `<w>/<a>/{processed_traces,ncu/full,ncu/cycles}`) by that depot's
  `accelsim-layout.sh` layer: the kernels in `H200/ML-DROPPED-KERNELS.md` are
  left out of `kernelslist.g` and their HW rows dropped too, so both sides pair
  one-to-one; `ncu.csv.gpc__cycles_elapsed.0` is ncu_cycles.0.csv in the long
  one-metric format. Not in it: `llama_inference_32k` (no traces), FA3
  (`benchmark_hopper_forward_varlen_prefill.py`: no traces). Like every
  static-depot trace, nothing updates them automatically.
- ~9K per-kernel jobs: `run_simulations.py --per-kernel --array` submits each
  app's kernels as Slurm job arrays (<= MaxArraySize = 1001 tasks), one queue
  record while pending, instead of ~9K sbatch calls against the cluster-wide
  MaxJobCount (10000). Tasks log as `<array>_<task>`; `job_status` uses
  `squeue -r`; `slurm.sim` names outputs by `ACCELSIM_JOB_ID` when set.
  Archived under `ml/H200-SASS` so it cannot collide with `ubench/H200-SASS`.
- `--allow-oom` (weekly only): the cluster does not enforce `--mem`
  (`TaskPlugin=task/affinity`, no `task/cgroup`), so Slurm never reports
  OUT_OF_MEMORY here: the node's OOM killer SIGKILLs the job (sacct FAILED,
  exit 137:0 or 0:9). `common.oom_killed` follows the slurm-monitor skill: not a
  segfault (signal 11 / 139) or assert (signal 6 / 134), and OUT_OF_MEMORY /
  NODE_FAIL, signal 9 / exit 137, an oom-kill line in the last 4 KB of the logs,
  or MaxRSS > ReqMem. Light: the monitor polls as before and only at the end
  runs one `sacct -X` for the failed jobs (exit 0 if all were OOM kills);
  get_stats checks only failed sims; MaxRSS (job steps) only for failed jobs.
  A killed job's output may never appear, so job_status waits at most 5 s for
  it instead of spinning forever.
- `get_stats --per-kernel-dirs` names each kernel column by its CUDA name and
  occurrence (`-K`), as the normal mode does, so plot-correlation can pair it.
  2026-10-19 check on the weekly's ML output: 8942 kernels of 20 apps paired, no
  KERNEL-MISMATCH; warp instructions 1.0% MAE, GPC cycles 14.6% MAE (kernel).
- `publish_depot` replaces a depot only with a finished trace + hardware-stat
  run; a failed tracer leaves the previous week's traces.
- The V100 tracer and apps build on the runner host; the trace and the
  hardware stats are two gpu-lock sruns, so PR Tracer-Tool runs can interleave.
- Measured (successful runs, Aug 14 to Sep 25 2026): Tracer-Hopper 2h45m to
  3h47m (13 runs), Tracer-Volta 5h41m to 5h50m with the build (3 runs). The
  remote `h200.sh` gets an 8h `timeout`; ssh/rsync keepalives (60 s x 5)
  because public run 38952168971 (Oct 16) hung silent for 8h22m.
- `CI_SANDBOX=<dir>` redirects every side-effect path (trace depots, the
  archive as a local bare repo, lastSuccess, ci-stats, the remote
  work dir) at once; setting one of the removed per-path knobs is an error.
- Every simulated app must have hardware stats (`NO-HW-REFERENCE` is a
  plot-correlation error), so PR #146's vLLM and the Spinlock runs are not in
  the weekly: neither had any. lavaMD/srad_v1 ran past the 12h monitor in
  PR #146's last green run (queue shared with the dropped paper job); left as
  is, expected to be fast on the new traces.

## Open

- Hopper has only the generic `"all"` correlation stats.
- The static depot (`accel-sim/H200`, dev's full tier) had no hardware stats
  for GPU_Microbenchmark_TMA and cutlass_tma_small; with `NO-HW-REFERENCE`
  fatal, dev's full tier needs them copied from the weekly's nightly depot.
