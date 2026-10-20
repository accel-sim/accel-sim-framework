#!/usr/bin/env python3
"""Publish pre-collected traces as GitHub release assets for get-accel-sim-traces.py.

Input, one directory per argument set:
  <src>/<workload>/<args>/<traces_subdir>/kernelslist.g + kernel-*.trace(z|g...)
  <src>/<workload>/<args>/<ncu_csv>        (optional Nsight Compute CSV)

Each argument set becomes one uncompressed tar (traces are already compressed)
laid out as
  <GPU>/<workload>/<args>/traces/...
  <GPU>/<workload>/<args>/ncu_stats.csv
so that run_simulations.py -T <dir>/<GPU> and plot-correlation.py -H <dir>/<GPU>
both work after extraction. Tars are split into parts below GitHub's 2 GiB asset
limit, uploaded with `gh release upload`, and described in manifest.json and a\nREADME.md with download instructions.

Re-running resumes: argument sets already recorded in <work>/meta are skipped.
"""

import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

PART_BYTES = 1900 * 1024 * 1024
assert PART_BYTES < 2 * 1024**3, "GitHub release assets must be smaller than 2 GiB"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Publisher:
    def __init__(self, o):
        self.o = o
        self.lock = threading.Lock()
        os.makedirs(os.path.join(o.work, "meta"), exist_ok=True)
        os.makedirs(os.path.join(o.work, "stage"), exist_ok=True)

    def gh_upload(self, paths):
        for attempt in range(5):
            r = subprocess.run(
                ["gh", "release", "upload", self.o.tag, "-R", self.o.repo, "--clobber", *paths],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                return
            print(f"upload retry {attempt}: {r.stderr.strip()[:200]}", flush=True)
        raise RuntimeError(f"upload failed: {paths}")

    def write_manifest(self):
        entries = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(self.o.work, "meta", "*.json")))]
        path = os.path.join(self.o.work, "manifest.json")
        with open(path, "w") as f:
            json.dump(dict(format=1, repo=self.o.repo, tag=self.o.tag, traces=entries), f, indent=1)
        readme = os.path.join(self.o.work, "README.md")
        with open(readme, "w") as f:
            f.write(open(self.o.readme).read().rstrip("\n") + "\n\n")
            f.write("| GPU | Workload | Arguments | Kernels | Size | Parts |\n|---|---|---|---|---|---|\n")
            for e in sorted(entries, key=lambda e: (e["gpu"], e["workload"], e["args"])):
                f.write(f"| {e['gpu']} | {e['workload']} | {e['args']} | {e['kernels']} | {e['size'] / 1e9:.2f} GB | {len(e['parts'])} |\n")
        self.gh_upload([path, readme])
        return entries

    def one(self, workload, args):
        o = self.o
        base = f"{o.gpu}__{workload}__{args}.tar"
        meta = os.path.join(o.work, "meta", base + ".json")
        if os.path.exists(meta):
            print(f"skip {base} (already published)", flush=True)
            return
        src = os.path.join(o.src, workload, args)
        tdir = os.path.join(src, o.traces_subdir)
        kernels = [l for l in open(os.path.join(tdir, "kernelslist.g")).read().split("\n") if l.startswith("kernel-")]
        files = sorted(f for f in os.listdir(tdir) if f.startswith("kernel-"))
        assert sorted(kernels) == files, f"{workload}/{args}: kernelslist.g lists {len(kernels)} kernels, directory has {len(files)} trace files"
        pre = f"{o.gpu}/{workload}/{args}"
        members = [f"{workload}/{args}/{o.traces_subdir}"]
        transforms = [f"s|^{workload}/{args}/{o.traces_subdir}|{pre}/traces|"]
        ncu = os.path.join(src, o.ncu_csv) if o.ncu_csv else None
        if ncu and os.path.exists(ncu):
            members.append(f"{workload}/{args}/{o.ncu_csv}")
            transforms.append(f"s|^{workload}/{args}/{o.ncu_csv}|{pre}/ncu_stats.csv|")
        elif ncu and o.require_ncu:
            sys.exit(f"{workload}/{args}: missing {ncu}")
        stage = os.path.join(o.work, "stage", base)
        os.makedirs(stage, exist_ok=True)
        for p in glob.glob(stage + "/*"):
            os.remove(p)
        tar = ["tar", "-C", o.src, "--dereference", "--sort=name", "-cf", "-"]
        tar += [f"--transform={t}" for t in transforms] + members
        cmd = (
            "set -o pipefail; "
            + " ".join(map(shlex.quote, tar))
            + f" | tee >(sha256sum | cut -d' ' -f1 > {shlex.quote(stage)}/full.sha256)"
            + f" | split -b {PART_BYTES} -d -a 3 - {shlex.quote(os.path.join(stage, base))}.part"
        )
        subprocess.run(["bash", "-c", cmd], check=True)
        parts = sorted(glob.glob(os.path.join(stage, base + ".part*")))
        assert parts and all(os.path.getsize(p) <= PART_BYTES for p in parts), f"{base}: bad split"
        plist = [dict(name=os.path.basename(p), size=os.path.getsize(p), sha256=sha256(p)) for p in parts]
        self.gh_upload(parts)
        entry = dict(
            gpu=o.gpu,
            workload=workload,
            args=args,
            path=pre,
            kernels=len(kernels),
            size=sum(p["size"] for p in plist),
            sha256=open(os.path.join(stage, "full.sha256")).read().strip(),
            parts=plist,
        )
        with open(meta, "w") as f:
            json.dump(entry, f, indent=1)
        for p in parts:
            os.remove(p)
        print(f"published {base}: {len(plist)} parts, {entry['size'] / 1e9:.2f} GB", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Root holding <workload>/<args>/")
    p.add_argument("--gpu", required=True, help="GPU name used as the top-level directory, e.g. H200")
    p.add_argument("--repo", required=True, help="GitHub repository, e.g. accel-sim/accel-sim-framework")
    p.add_argument("--tag", required=True, help="Release tag; the release must already exist")
    p.add_argument("--work", required=True, help="Scratch directory for tar parts and per-set metadata")
    p.add_argument("--traces_subdir", default="traces", help="Trace directory inside each <workload>/<args> (default: traces)")
    p.add_argument("--ncu_csv", default="", help="Nsight Compute CSV inside each <workload>/<args>, shipped as ncu_stats.csv")
    p.add_argument("--require_ncu", action="store_true", help="Fail if an argument set has no NCU CSV")
    p.add_argument("--jobs", type=int, default=3, help="Argument sets packed and uploaded in parallel")
    p.add_argument(
        "--readme",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "trace-release-README.md"),
        help="Template uploaded as the release's README.md, with a table of the sets appended",
    )
    p.add_argument("only", nargs="*", help="Optional <workload>/<args> subset")
    o = p.parse_args()

    sets = sorted(
        (k.split("/")[-4], k.split("/")[-3])
        for k in glob.glob(os.path.join(o.src, "*", "*", o.traces_subdir, "kernelslist.g"))
    )
    if o.only:
        sets = [s for s in sets if f"{s[0]}/{s[1]}" in o.only]
        missing = set(o.only) - {f"{a}/{b}" for a, b in sets}
        assert not missing, f"not found under {o.src}: {sorted(missing)}"
    assert sets, f"no <workload>/<args>/{o.traces_subdir}/kernelslist.g under {o.src}"
    pub = Publisher(o)
    print(f"{len(sets)} argument sets -> {o.repo} release {o.tag}", flush=True)
    with ThreadPoolExecutor(o.jobs) as ex:
        list(ex.map(lambda s: pub.one(*s), sets))
    entries = pub.write_manifest()
    print(f"manifest.json: {len(entries)} argument sets, {sum(e['size'] for e in entries) / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
