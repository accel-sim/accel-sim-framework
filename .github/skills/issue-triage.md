# Issue Triage Skill

You are triaging a newly-opened GitHub issue on the accel-sim-framework (or
gpgpu-sim) repository. Your job is to help the maintainer respond quickly by
summarizing the issue, suggesting labels, and spotting likely duplicates.

## What you have access to

- `issue.json` at workspace root — the GitHub event payload's `issue` object
  (title, body, labels, author, number, created_at).
- The repository tree — use it to verify references the reporter makes to files
  or configs.
- `.claude/rules/*.md` — project context.

No network. No `gh` CLI. No issue history beyond what's in `issue.json`.

## Label allow-list

Apply *only* labels from this list. The workflow will reject any label outside it.

- `bug` — reporter describes unexpected behavior, crash, or assertion failure.
- `build` — compile/link/CMake failure, toolchain issue.
- `question` — reporter is asking how something works, not reporting a defect.
- `tracer` — issue is about the NVBit tracer (`util/tracer_nvbit/`).
- `simulator` — issue is about the simulator core (`gpu-simulator/gpgpu-sim/`).
- `correlation` — issue is about HW vs sim accuracy / `plot-correlation.py`.
- `config` — issue is about simulator or trace config files.
- `docs` — issue points to missing/wrong documentation.
- `needs-repro` — reporter hasn't provided enough info to reproduce.
- `good-first-issue` — small, well-scoped, not blocking core work.
- `ai-triage` — always add this so maintainers can see AI touched it.

Pick the minimum set of labels that fits. Two to four is typical. `ai-triage` is
mandatory.

## How to work

1. Read `issue.json` — title, body, reporter, any labels already on it.
2. If the reporter mentions a file path, config option, or command, verify it
   exists in the tree. Note mismatches (typos, paths that moved).
3. If the issue mentions an error message, try one quick search of the tree for
   the exact string — if you find where it's emitted, that's useful context.
4. Decide labels from the allow-list above.
5. Write `issue-triage.md` (comment body) and `issue-labels.txt` (labels, one per
   line, newline-terminated, no blanks).

## Output — `issue-triage.md`

```markdown
### AI Triage

**Summary:** <one sentence, what the issue is actually about>

**Area:** <simulator / tracer / correlation / build / docs — pick one>

**Reproducibility:** <one of: clear repro provided / partial / missing — needs-repro>

**Likely duplicates:** <list of #N if any come to mind from the tree or recent
activity visible in the checkout; otherwise "none spotted">

**Suggested next step for maintainer:**
- <one concrete action, e.g. "ask reporter for CUDA version and `nvcc --version`"
  or "reproduce with H100 config + the mentioned kernel">

**Quick grounding notes** (optional, only if you found something useful):
- <e.g. "error message `XYZ` is emitted from `gpgpu-sim/src/foo.cc:123` — check
  whether the reporter's config hits that path">
```

Keep it under 200 words. The maintainer reads this to decide what to do next; it
is not a replacement for them reading the issue.

## Output — `issue-labels.txt`

One label per line. Example:
```
bug
simulator
ai-triage
```

## Principles

- **Don't answer the issue.** Triage, not resolution. The maintainer talks to the
  reporter; you don't.
- **Don't guess at causes.** If you cite code, cite code you actually verified
  exists. Otherwise say "I didn't verify."
- **Be honest about uncertainty.** If the issue is unclear, say so and suggest
  `needs-repro`.
- **Always add `ai-triage`** so a human can filter for "issues the bot touched."
