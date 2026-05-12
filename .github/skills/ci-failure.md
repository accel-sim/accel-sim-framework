# CI Failure Analysis Skill (local mode)

You are analyzing a failed CI run from inside the CI runner's workspace. You do NOT have
`gh` CLI or network access to GitHub. Everything you need is on the local filesystem.

**Do NOT worry about token usage or context length. Perform a thorough, maximum-effort
investigation.** Read full log outputs, examine complete error files, trace through source
code, check git history, and cross-reference. The goal is to give the developer a complete
picture so they never need to open the raw CI logs.

## Anchor on the real error first

Before anything else, find the literal message that caused the step to exit non-zero. The
last error/abort/`exit` line printed before `Process completed with exit code 1` is your
anchor — every claim in your report must ground back to it.

- If you cannot locate that string anywhere in the workspace or the runner step log, say
  so explicitly in the report. **Do not invent one by grepping recent diffs for
  error-shaped strings** — many sources contain `printf("Error: ...")` lines that never
  fire on this run.
- Before naming any source line as the cause, verify the exact error text appears in the
  actual run output. If it doesn't appear, it isn't the cause.

## Where to look

- **Simulation logfiles**: `util/job_launching/logfiles/` — the latest `sim_log.*.txt` and `failed_job_log_*.txt`
- **Simulation output**: `sim_run_<cuda_version>/<app>/<args>/<config>/` — `.o<jobid>` (stdout) and `.e<jobid>` (stderr)
- **Per-kernel output**: `sim_run_<cuda_version>/<app>/<args>/<config>/per-kernel/kernel-<N>/` — same `.o` and `.e` files
- **Slurm job status**: use `sacct` for job states, memory, elapsed time
- **Stats CSVs**: root of workspace and `statistics-archive/`
- **Git info**: `git log`, `git show`, `git diff` in the root repo and in `gpu-simulator/gpgpu-sim/` (separate repo)
- **Build log**: `build.log` in workspace root (tee'd from the Build step) — full compiler/linker output when the build failed
- **CMake build dir**: `gpu-simulator/build/` — `CMakeFiles/CMakeError.log`, `CMakeOutput.log`, etc.
- **CI scripts**: `.github/scripts/main-sass.sh`, `.github/scripts/main-tracer.sh`,
  `.github/scripts/hopper-weekly.sh`, `.github/scripts/h200.sh`, and
  `.github/scripts/lib/common.sh`. Failures often happen in post-simulation orchestration
  (rsync, ln, git push, mv) rather than in the simulator itself.

## Analysis steps

1. **Find the latest logfile and failed-job log**:
   ```bash
   ls -t util/job_launching/logfiles/sim_log.*.txt | head -1
   ls -t util/job_launching/logfiles/failed_job_log_*.txt | head -1
   ```

2. **Parse the failed job log** to identify which benchmarks failed:
   ```bash
   cat <failed_job_log>
   ```
   Lists job IDs, benchmarks, configs, and failure status.

3. **For each failed job**, read the actual error from the simulation directory:
   ```bash
   sacct -j <JOBID> --format=WorkDir%300 -n | head -1
   cat <workdir>/*.e<JOBID>    # stderr — assertions/crashes
   tail -20 <workdir>/*.o<JOBID>   # stdout — deadlocks, last kernel
   ```

4. **Check Slurm job details** (OOM, timeouts):
   ```bash
   sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,Elapsed,NodeList -n
   ```

5. **Find stuck/running jobs** (cross-reference logfile vs sacct):
   ```bash
   awk '{print $2}' <logfile> | xargs -I{} sacct -j {} --format=JobID,JobName%40,State -n | grep RUNNING
   ```

## Failure classification

- **Build failure** — compiler/linker/CMake errors. Extract exact error + file:line.
- **Simulation assertion / crash** — `Assertion ... failed`, `SIGABRT`, `SIGSEGV`, `=== CRASH ===`. Extract message, file:line, backtrace, which benchmark+config triggered it.
- **Functional test failure (FUNC_TEST_FAILED)** — simulation completed but output was wrong. Check if the app printed `FAILED` (mismatch) vs the simulator crashed.
- **Timeout / Cancelled** — identify which specific benchmark(s) were stuck. Jobs listed in the logfile but missing from the monitor's completion output are the stuck ones. Also inspect `failed_job_log`.
- **Deadlock** — `ERROR ** deadlock detected`. Extract kernel name, cycle, stuck cores.
- **OOM Kill** — `Killed` in stderr. Compare `sacct` MaxRSS vs ReqMem.
- **Format check failure** — code wasn't formatted.
- **Infrastructure / flaky failure** — Docker, network, runner issues. Suggest re-run.

## Root-cause analysis

0. **Confirm where the failure actually is.** Check which sub-steps in the failing job
   succeeded (`Run SASS Simulations`, `Archive Stats`, `Correlate Ubench`, etc.). If the
   simulation and archive steps both succeeded, the failure is in post-simulation
   orchestration — look at the `.github/scripts/*.sh` stage that ran last, not at the
   simulator source.
1. **Identify the failing commit**: `git log -1 --oneline`
2. **See what changed**:
   ```bash
   git log --oneline -10
   git show HEAD --stat
   git diff HEAD~1..HEAD
   ```
   Use `gu-simulator/gpgpu-sim/` for gpgpu-sim changes, root repo for accel-sim changes.
3. **Trace the crash to code**: grep for the assertion text in the source, read surrounding
   code, cross-reference with the commit diff to find which change introduced the bug.
4. **Check if pre-existing**: look at prior output files in the same sim directory.

## Notes on the CI

- SASS tests use `QV100-SASS`, `A100-SASS`, `H200-SASS` configs.
- PTX / CMake tests use `TITANV`, `QV100`, `RTX2060`, `RTX3070`, `TITANV-LOCALXBAR`.
- `FUNC_TEST_FAILED` = simulation ran but output was wrong.
- `FUNC_TEST_FAILED, ASSERT` = simulator crashed with an assertion.
- `COMPLETE_ERR_FILE_HAS_CONTENTS` = simulation completed but wrote to stderr. Read the
  error file to decide if it's a warning or a real error.
- Cancelled jobs are usually side-effects — focus on the actual root failure.
- The monitor's `-T <hours>` timeout cancels the whole CI job if one sim exceeds it.

## Output

**Write your final report as GitHub-flavored markdown to `ci-failure-report.md` in the
workspace root.** Do not print the final report to stdout; write only to the file.

Use this structure:

```markdown
## CI Failure Analysis

**Branch**: <branch> @ <short_sha>
**Trigger**: <push|pull_request|workflow_dispatch>

### Summary
<One sentence: what failed and why.>

### Failed Job(s)
- **<job_name>**: <failure category>

### Root Cause
<Detailed explanation, referencing:>
- File and line number of the crash/error
- The commit that introduced the issue (if identifiable)
- What the code does and why it fails
- Which benchmark/config exposed the bug

### Relevant Code
<Problematic code snippet or diff hunk.>

### Suggested Fix
<Concrete suggestion.>
```
