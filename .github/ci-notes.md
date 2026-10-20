# Accel-Sim CI (for the AI failure and result analysis)

- `main.yml`: check-format, then SASS-Simulation (build, ubench simulations
  on `QV100-SASS`, `A100-SASS`, `H200-SASS`, archive stats, correlate against
  hardware), PTX-Simulation (`TITANV`, `QV100`, `RTX2060`, `RTX3070`,
  `TITANV-LOCALXBAR`), Tracer-Tool (build the NVBit tracer, trace, test the
  new traces). `hopper-weekly.yml`: traces on a remote H200 over VPN, then
  SASS simulation and correlation for H100, A100 and V100.
- gpgpu-sim is cloned at build time (`.github/scripts/clone-gpgpusim.sh`, a
  same-named branch of the PR author's fork if there is one, else `dev`): a
  simulator failure may come from that repo, not this one.
- Simulations run as slurm jobs; the monitor prints each job's status.
  `FUNC_TEST_FAILED` = the simulation ran but its output was wrong;
  `FUNC_TEST_FAILED, ASSERT` = the simulator hit an assertion;
  `COMPLETE_ERR_FILE_HAS_CONTENTS` = completed but wrote to stderr (a warning
  or a real error).
- The monitor's `-T <hours>` timeout cancels the whole job if one simulation
  exceeds it. Cancelled jobs are usually side effects of another failure.
- Failures after the simulations succeeded are often in the orchestration
  scripts (`.github/scripts/*.sh`, `lib/common.sh`: rsync, ln, git push to
  the statistics archive, mv), not in the simulator.
