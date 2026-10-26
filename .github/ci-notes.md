# Accel-Sim CI (for the AI failure and result analysis)

- `main.yml`: check-format, then SASS-Simulation, PTX-Simulation (`rodinia_2.0-ft`
  and `GPU_Microbenchmark_CI`, on dev the full `GPU_Microbenchmark`, on
  `QV100-PTX`, `A100-PTX`), Tracer-Tool (build the NVBit tracer, trace, trace
  again on GPUs 3,7 with run_hw_trace.py's worker pools and compare
  (`compare-traces.py`), test the new traces), SASS-Simulation-Docker.
  SASS-Simulation has two tiers (`CI_TIER`, `main-sass.sh`): smoke on PRs
  (`GPU_Microbenchmark_CI`, `rodinia_2.0-ft` on `QV100-SASS` and `H200-SASS`,
  `GPU_Microbenchmark_TMA_CI`, `cutlass_tma_CI`, `hopper_features_CI` on
  `H200-SASS`; no correlation, no archive) and full on
  dev (`GPU_Microbenchmark`, `rodinia_2.0-ft` on `QV100-SASS` and `H200-SASS`, plus `cutlass_tma_small`, `GPU_Microbenchmark_TMA` on H200;
  correlated and archived).
- `weekly.yml` (`weekly.sh <stage>`): Tracer-H200 (traces and hardware stats
  on a remote H200 over VPN) then SASS-H200 (`H200-SASS`: rodinia, ubench, TMA,
  cutlass_tma_small); Tracer-V100 (on tgrogers-dgx) then SASS-V100
  (`QV100-SASS`: rodinia, ubench); SASS-H200-ML (`H200-SASS` on the hosted
  ML traces of the static H200 depot, never traced here). Each SASS job simulates,
  correlates, archives.
- gpgpu-sim is cloned at build time (`clone_gpgpusim` in `lib/common.sh`: the
  same-named branch of the code's gpgpu-sim fork if it has one, else upstream
  `dev`, also when there is no such fork; any other clone or fetch failure
  fails the build): a simulator failure may come from that repo, not this one.
- Simulations run as slurm jobs; the monitor prints one summary line per poll
  and, at the end, the failed jobs' rows (their output tails are in
  `util/job_launching/logfiles/failed_job_log_*.txt`). Every
  SASS run except SASS-Simulation-Docker is per kernel (`--per-kernel --array
  -M 10G`): one Slurm array task per kernel, job ids `<array>_<task>`, run dirs
  `.../<config>/per-kernel/<kernel>/`.
  `FUNC_TEST_FAILED` = the simulation ran but its output was wrong;
  `FUNC_TEST_FAILED, ASSERT` = the simulator hit an assertion;
  `COMPLETE_ERR_FILE_HAS_CONTENTS` = completed but wrote to stderr (a warning
  or a real error); `SIM_ERROR` = GPGPU-Sim printed `ERROR **`;
  `SLURM_<STATE>(<exit>)` = Slurm ended the job other than COMPLETED (e.g.
  `SLURM_FAILED(1:0)`: the simulator exited 1).
- The monitor's `-T <hours>` timeout cancels the whole job if one simulation
  exceeds it. Cancelled jobs are usually side effects of another failure.
- A simulation job that fails or is cancelled scancels its own Slurm jobs
  (comment `gh-run:<run>.<attempt>:<job>`) before the runner is reused: the
  next job's checkout deletes the run dirs, and sims still running then
  segfault (`Stale file handle`).
- PTX-Simulation links the rodinia data from the lab's shared copy
  (`/scratch/.../for-sharing/gpgpu-sim-appdata/data_dirs`); no download.
- After the monitor, `get_stats.py --strict` fails the simulate step if any
  simulation has no output or no GPGPU-Sim terminating string
  (`ERROR --strict: <config> <app>: <reason>`).
- Correlation output per config is in `ci-stats/<GPU>-<config>/correl.txt`:
  `NO-HW-REFERENCE` = a simulated app without hardware stats (an error: the
  hardware depot lacks that app); `KERNEL-MISMATCH` = sim and hardware kernels
  of an app do not all pair by name; `ERROR:` lines fail the step.
- Failures after the simulations succeeded are often in the orchestration
  scripts (`.github/scripts/*.sh`, `lib/common.sh`: rsync, ln, git push to
  the statistics archive, mv), not in the simulator.
