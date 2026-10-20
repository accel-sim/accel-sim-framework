---
description: CI statistics, correlation and archive contracts
paths:
  - ".github/**"
  - "util/plotting/merge-stats.py"
  - "util/plotting/plot-correlation.py"
  - "util/plotting/correl_mappings.py"
  - "util/job_launching/get_stats.py"
---

# CI stats pipeline

Read `docs/ci-stats.md` before redesigning any of this.

- Never archive or correlate from a failed run: archive/correlate steps rely on
  the default `success()`; never `always()`, `!cancelled()` or `|| log_error`.
- Each script has one table of what it runs (`main-sass.sh` RUNS, `weekly.sh`
  gpu_vars); verify, correlate and archive take their configs from it.
- CI calls `get_stats.py --strict`; a missing or unterminated sim must fail.
  Only the weekly passes `--allow-oom` (monitor, get_stats): a failed job
  `common.oom_killed` classifies as killed for memory is a warning, checked once
  at the end. Without the flag nothing changes; other launchers: no effect.
- Archive layout: `ubench/<GPU>-<config>/runs/<time>_<run>-<attempt>_<sha>.csv`
  is never rewritten; `history.csv` is regenerated from `runs/`. Name dirs by
  the config (`H200-SASS`), never by a GPU alias (`hopper-h100`).
- Push the archive without `-f`: `archive_runs` refetches and restages on a
  rejected push. SASS-H200 and SASS-V100 share one weekly branch.
- `merge-stats.py` output must contain every app, kernel, stat and config of
  every input; it exits otherwise.
- `correl_mappings.config_maps` holds exact device names (sets of strings, not
  `set("...")`); plot-correlation needs exactly one match per config.
- Kernels pair by (demangled base name, occurrence), not position.
- PRs never write the archive.
- No run-to-run stat comparison: CI correlates and archives, nothing more.
- Email only from dev pushes and weekly (the runs that correlate). PR runs
  send none: keep `send_ci_email.py` behind the push-to-dev check in `main.yml`.
- dev runs use a per-commit concurrency group and are never cancelled; every
  dev commit is archived and correlated. Other refs cancel their previous run.
- Weekly: Tracer-<GPU> -> SASS-<GPU> per GPU, plus SASS-H200-ML (hosted traces,
  never traced, archive group `ml`); one `weekly.sh` stage per step; one run
  per repo at a time (`concurrency`). No A100.
- `--array` (with `--per-kernel`) logs tasks as `<array>_<task>`; keep job ids
  strings and `squeue -r` in job_status. Without `--array` nothing changes.
- A trace depot is replaced only by `publish_depot` of a finished run.
- `CI_SANDBOX=<dir>` is the only path override; do not add per-path env knobs.
- Every simulated app needs hardware stats: `NO-HW-REFERENCE` fails
  plot-correlation. Do not simulate a suite the depot has no stats for.
