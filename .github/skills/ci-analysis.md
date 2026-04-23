# CI Success Analysis Skill (local mode)

You are analyzing a successful CI run from inside the CI runner's workspace. You do NOT
have `gh` CLI or network access to GitHub. Everything you need is on the local filesystem.

**Do NOT worry about token usage or context length. Perform a thorough, maximum-effort
investigation.** Read full correlation CSVs, parse every kernel's error, compare against
previous runs. The goal is to give the developer a complete picture of how simulation
accuracy changed — what regressed, what improved, what silently broke — so they never
need to dig through CI artifacts themselves.

## Where to look

- **Correlation output**: `util/plotting/correl-html/` — `*.kernel.raw.csv`, `*.app.raw.csv`, `*.kernel.txt`, `*.apps.txt`
- **Statistics archive**: `statistics-archive/ubench/` — current and previous merged CSVs, `-latest.csv` files
- **Simulation logfiles**: `util/job_launching/logfiles/` — latest `sim_log.*.txt` and `failed_job_log_*.txt`
- **Simulation output**: `sim_run_<cuda_version>/<app>/<args>/<config>/` — `.o<jobid>` / `.e<jobid>`
- **Slurm job status**: `sacct` for elapsed time, MaxRSS
- **Git info**: `git log`, `git show`, `git diff` on root repo and `gpu-simulator/gpgpu-sim/`
- **CI scripts**: `.github/scripts/hopper-weekly.sh` and `.github/scripts/lib/common.sh`

## Correlation CSV format

Per-kernel raw CSVs in `util/plotting/correl-html/` are named
`<stat>.<configs>.kernel.raw.csv`. The first line is a summary:

```
H200-SASS (N apps, M kernels (X < 1% Err, Y under, Z over)) [Correl=0.XXXX Err=XX.XX%]
```

Remaining rows:

```
Name,Hardware,Simulator,Sim/HW
app/args--kernel_name (Err=X%,HW-Range=+Y%/-Z%),hw_val,sim_val,ratio
```

Key stats to look at:
- `gpc_cycles` — primary cycle-accuracy metric
- `dram-read-transactions` — memory-system accuracy
- `l2-read-hits`, `l2-write-hits` — cache accuracy
- `gpu_tot_sim_insn` — instruction count accuracy

## Analysis steps

### 1. Find the data

```bash
ls -t util/job_launching/logfiles/sim_log.*.txt | head -1
ls util/plotting/correl-html/*.kernel.raw.csv
ls statistics-archive/ubench/
```

### 2. Current-run accuracy

For each config's `*.kernel.raw.csv`, extract from the summary line:
- correlation coefficient
- mean error
- number of kernels within 1% error

### 3. Compare vs previous run

The statistics-archive holds the previous merged CSVs (`-latest.csv`). Diff:
- Correlation coefficient change (e.g. 0.9955 → 0.9961)
- Mean error change (e.g. 16.44% → 14.20%)
- Kernels that moved in/out of `<1% error`

Per-kernel regression flags:
- Error increased by more than 10 percentage points vs previous
- Sim/HW ratio flipped direction (under → over or vice versa)
- A kernel that was within 5% error is now beyond 20%

### 4. Detect silent failures

A passing CI with missing or broken kernels is the most dangerous case.

```bash
# Any non-COMPLETED Slurm jobs from this run?
awk '{print $2}' <logfile> | xargs -I{} sacct -j {} --format=JobID,State -n | grep -v COMPLETED
```

- Kernels present in previous CSV but missing now → likely deadlocked/crashed
- New kernels not in previous → new benchmarks or traces (note in report)
- Kernels with `NA` values → partial failures

For each failed/missing kernel, check `.e<jobid>` for assertions/OOM and `.o<jobid>` for
deadlock messages. Extract the kernel name from the trace filename.

### 5. Simulation performance

From the stats CSV and logfile:
- `gpgpu_simulation_rate` (cycles/sec)
- `gpgpu_silicon_slowdown`
- Flag unusually slow benchmarks (possible near-infinite loops)

```bash
awk '{print $2}' <logfile> | xargs -I{} \
  sacct -j {} --format=JobID,JobName%40,State,MaxRSS,Elapsed -n
```
Flag jobs near time limit or with high MaxRSS.

## Report

Write your final report as GitHub-flavored markdown to `ci-analysis-report.md` in the
workspace root. Do not print it to stdout.

Use this structure:

```markdown
## CI Simulation Analysis

**Branch**: <branch> @ <short_sha>
**Configs**: <list>

### Quick Summary
- X/Y benchmarks completed successfully
- N kernels show >10pp accuracy regression
- M simulations silently failed (deadlock/crash)

### Accuracy Changes (vs previous run)

#### Cycle Correlation (gpc_cycles)
| Config | Prev Correl | Curr Correl | Prev Err | Curr Err | Kernels <1% |
|--------|-------------|-------------|----------|----------|-------------|
| H200-SASS | 0.9955 | 0.9961 | 16.44% | 14.20% | 36 → 52 |

#### Largest Regressions (by error increase)
| Kernel | Prev Err | Curr Err | Change | Prev Sim/HW | Curr Sim/HW |
|--------|----------|----------|--------|-------------|-------------|
| app--kernel | -5.2% | -28.1% | -22.9pp | 0.95 | 0.72 |

#### Largest Improvements
| Kernel | Prev Err | Curr Err | Change |
|--------|----------|----------|--------|

### Silent Failures

#### Deadlocked Kernels (N)
| App | Kernel | Deadlock Cycle |
|-----|--------|----------------|

#### Missing Kernels (vs previous run)
Kernels that were in the previous correlation but are absent now (likely deadlocked or
crashed):
- `app/args--kernel_name`

#### Crashed Simulations
| App | Kernel | Error |
|-----|--------|-------|

### Simulation Performance
| Config | Avg Rate (cyc/s) | Slowest Benchmark | Time |
|--------|------------------|-------------------|------|

### Memory Issues
Jobs that were OOM-killed or hit memory limits:
| App | Kernel | MaxRSS | ReqMem |
|-----|--------|--------|--------|
```

## Principles for a good report

- **Be specific** — name the exact kernels, apps, error values.
- **Highlight change** — developers care about regressions, not absolute numbers.
- **Flag silent failures** — a "passing" CI with missing kernels is worse than a visible failure.
- **Don't overwhelm** — top 10 regressions, top 10 improvements. Omit sections entirely if empty.
- **Actionable** — for each problem, suggest what to look at (e.g. "kernel X regressed — check if the L2 config change affected it").
