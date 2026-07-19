## Summary

<!-- 2-4 sentences: what does this PR do, and why does it matter to someone
who has never seen the bug/feature you're addressing? Lead with impact,
not mechanism. -->

## Motivation / root cause

<!-- For bug fixes: what's the concrete failure, and what's the underlying
cause? Show the actual error/output, not a paraphrase — reviewers should be
able to verify your diagnosis against the evidence, not just trust it.
For features: what gap does this fill, and why now?

Bad:  "The parser was broken for newer CUDA."
Good: "ptxas 12.8 emits used N barriers and `N bytes cumulative stack
       size` in its resource-usage summary line. GPGPU-Sim's ptxinfo.l/.y
       grammar predates both fields, so every kernel compiled under
       CUDA 11+ hits GPGPU-Sim: *** exit detected *** before running a
       single cycle. Confirmed via [command] on [corpus]." -->

## Changes

<!-- Bullet list, one line per file or logical change. Link to the specific
lines for anything non-obvious. Group by "what" not "which file" if the
diff touches many files for one reason. -->

-
-

## Evidence (before / after)

<!-- The single most important section for anything claiming to fix a bug
or improve accuracy. Reviewers should not have to run your fix to believe
it works — show them.

- Before: exact error/output, or a stats table/CSV showing the broken state
- After: same command, same corpus, showing the fix
- If numeric (correlation, MAPE, cycle counts): a before/after table beats
  a paragraph -->

*Before:*


*After:*


## Test plan

<!-- Concrete, runnable, checked off before you open the PR — not aspirational.
Include exact commands so a reviewer can reproduce your result, not just
your conclusion. -->

- [ ] ./format-code.sh clean (required — CI gate)
- [ ] ./short-tests.sh (or short-tests-cmake.sh) green
- [ ] <!-- targeted repro: exact command + expected output -->
- [ ] <!-- any new standalone test added by this PR: how to build/run it -->

## Compatibility / backward compatibility

<!-- Does this change behavior for existing configs/toolchains/apps that
were previously working? If the change is additive (old paths still work,
new paths are new alternatives), say so explicitly — reviewers will assume
the riskier case unless told otherwise. -->

## Limitations / known gaps

<!-- What did you deliberately NOT cover, and why? A stated limitation is a
sign of a well-scoped PR; an implied one (found later in review) costs a
review round-trip. E.g.: "Verified against the ptxas output shapes present
in [corpus]; other CUDA-version/kernel-shape combinations not exercised
here are not guaranteed." -->

## Related

<!-- Linked issues, related PRs, upstream discussions. Fixes #123 /
Closes #123 if applicable so GitHub auto-links. -->

---

<!-- Reviewer checklist — leave for the reviewer, don't fill yourself. -->
### For reviewers
- [ ] 
- [ ] 
- [ ] 
