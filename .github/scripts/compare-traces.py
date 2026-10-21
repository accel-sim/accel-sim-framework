#!/usr/bin/env python3
"""Check that two run_hw_trace.py output trees hold the same traces, app by app.

    compare-traces.py <baseline hw_run/traces/device-X/<cuda>> <other .../device-Y/<cuda>>

Two tracings of the same app differ in things that change on every run: the CUDA
context pointer in file names and kernelslist.g (kernel-1-ctx_0x55...tracez) and the
device addresses of the Memcpy lines. Both are masked. What must match:
- the set of app dirs and, in each, the set of files in traces/;
- kernelslist.g (kernel order, Memcpy type and size);
- stats_ctx_* (per-kernel instruction counts).
The kernel trace files themselves are not compared byte for byte; two serial runs
of the same app already differ there.
Exits 0 when everything matches.
"""
import os
import re
import sys

CTX = re.compile(rb"ctx_0x[0-9a-f]+")
MEMCPY_ADDR = re.compile(rb"^(Memcpy\w*),0x[0-9a-f]+,", re.M)


def norm_name(f):
    return CTX.sub(b"ctx", f.encode()).decode()


def norm_content(path):
    with open(path, "rb") as f:
        return MEMCPY_ADDR.sub(rb"\1,ADDR,", CTX.sub(b"ctx", f.read()))


# {app dir relative to root: {normalised file name: path}} for every traces/ dir
def app_dirs(root):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "traces":
            out[os.path.relpath(os.path.dirname(dirpath), root)] = {
                norm_name(f): os.path.join(dirpath, f) for f in filenames
            }
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = app_dirs(sys.argv[1]), app_dirs(sys.argv[2])
    ok = bool(a)
    if not a:
        print(f"no traces/ dirs under {sys.argv[1]}")
    for x in sorted(set(a) - set(b)):
        ok = False
        print(f"MISSING  {x}: only in the baseline")
    for x in sorted(set(b) - set(a)):
        ok = False
        print(f"EXTRA    {x}: only in {sys.argv[2]}")
    for app in sorted(set(a) & set(b)):
        fa, fb = a[app], b[app]
        problems = []
        if set(fa) != set(fb):
            problems.append(f"file sets differ: {sorted(set(fa) ^ set(fb))[:6]}")
        if "kernelslist.g" not in fa:
            problems.append("no kernelslist.g")
        for f in sorted(fa):
            if f == "kernelslist.g" or f.startswith("stats_ctx"):
                if f in fb and norm_content(fa[f]) != norm_content(fb[f]):
                    problems.append(f"{f} differs")
        if problems:
            ok = False
            print(f"DIFFER   {app}: " + "; ".join(problems))
        else:
            print(f"ok       {app} ({len(fa)} files)")
    print(f"{'PASS' if ok else 'FAIL'}: {len(set(a) & set(b))} apps in both trees")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
