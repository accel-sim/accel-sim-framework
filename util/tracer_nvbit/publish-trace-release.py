#!/usr/bin/env python3
"""Publish pre-collected traces as GitHub release assets for get-accel-sim-traces.py.

Input is the standard Accel-Sim layout under --src:
  <src>/<trace_root>/<workload>/<args>/traces/kernelslist.g + kernel-* trace files
  <src>/<hw_root>/<workload>/<args>/*.csv*          (Nsight Compute stats, optional)
with --trace_root traces/device-0/<cuda> and --hw_root device-0/<cuda> by default.

Each argument set becomes one uncompressed tar (traces are already compressed)
holding exactly the kernels listed in its kernelslist.g, prefixed with the GPU
name, e.g.
  H200/traces/device-0/12.8/<workload>/<args>/traces/...
  H200/device-0/12.8/<workload>/<args>/ncu.csv.nsight
Extracted under hw_run/, that is run_simulations.py -T hw_run/H200/traces/device-0/12.8
and plot-correlation.py -H hw_run/H200. Tars are split below GitHub's 2 GiB asset
limit, uploaded with `gh release upload`, and described in manifest.json and a
README.md with download instructions.

Re-running resumes: argument sets already recorded in <work>/meta are skipped.
"""

import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
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
            json.dump(dict(format=2, repo=self.o.repo, tag=self.o.tag, traces=entries), f, indent=1)
        readme = os.path.join(self.o.work, "README.md")
        with open(readme, "w") as f:
            f.write(open(self.o.readme).read().rstrip("\n") + "\n\n")
            f.write("| GPU | Workload | Arguments | Kernels | Size | Parts |\n|---|---|---|---|---|---|\n")
            for e in sorted(entries, key=lambda e: (e["gpu"], e["workload"], e["args"])):
                f.write(
                    f"| {e['gpu']} | {e['workload']} | {e['args']} | {e['kernels']} | {e['size'] / 1e9:.2f} GB | {len(e['parts'])} |\n"
                )
        self.gh_upload([path, readme])
        return entries

    def one(self, workload, args):
        o = self.o
        base = f"{o.gpu}__{workload}__{args}.tar"
        meta = os.path.join(o.work, "meta", base + ".json")
        if os.path.exists(meta):
            print(f"skip {base} (already published)", flush=True)
            return
        tdir = os.path.join(o.trace_root, workload, args, "traces")
        hdir = os.path.join(o.hw_root, workload, args)
        klist = open(os.path.join(o.src, tdir, "kernelslist.g")).read().split("\n")
        kernels = [l.strip() for l in klist if l.startswith("kernel-")]
        assert len(set(kernels)) == len(kernels), f"{workload}/{args}: duplicate kernel in kernelslist.g"
        members = [os.path.join(tdir, "kernelslist.g")]
        for k in kernels:
            assert os.path.isfile(os.path.join(o.src, tdir, k)), f"{workload}/{args}: kernelslist.g lists {k}, file missing"
            members.append(os.path.join(tdir, k))
        hw = sorted(glob.glob(os.path.join(o.src, hdir, "*.csv*")))
        if o.require_hw:
            assert hw, f"{workload}/{args}: no hardware CSV in {os.path.join(o.src, hdir)}"
        members += [os.path.relpath(h, o.src) for h in hw]
        stage = os.path.join(o.work, "stage", base)
        os.makedirs(stage, exist_ok=True)
        for p in glob.glob(stage + "/*"):
            os.remove(p)
        listing = os.path.join(stage, "members.txt")
        with open(listing, "w") as f:
            f.write("\n".join(members) + "\n")
        tar = ["tar", "-C", o.src, "--dereference", "-cf", "-", f"--transform=s|^|{o.gpu}/|", "-T", listing]
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
            trace_dir=f"{o.gpu}/{tdir}",
            hw_dir=f"{o.gpu}/{hdir}" if hw else None,
            trace_root=f"{o.gpu}/{o.trace_root}",
            hw_root=f"{o.gpu}/{o.hw_root}",
            kernels=len(kernels),
            size=sum(p["size"] for p in plist),
            sha256=open(os.path.join(stage, "full.sha256")).read().strip(),
            parts=plist,
        )
        with open(meta, "w") as f:
            json.dump(entry, f, indent=1)
        for p in parts:
            os.remove(p)
        print(f"published {base}: {len(kernels)} kernels, {len(plist)} parts, {entry['size'] / 1e9:.2f} GB", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Root holding <trace_root>/ and <hw_root>/")
    p.add_argument("--gpu", required=True, help="GPU name used as the top-level directory, e.g. H200")
    p.add_argument("--repo", required=True, help="GitHub repository, e.g. accel-sim/accel-sim-framework")
    p.add_argument("--tag", required=True, help="Release tag; the release must already exist")
    p.add_argument("--work", required=True, help="Scratch directory for tar parts and per-set metadata")
    p.add_argument("--trace_root", default="traces/device-0/12.8", help="Trace tree under --src")
    p.add_argument("--hw_root", default="device-0/12.8", help="Hardware stats tree under --src")
    p.add_argument("--require_hw", action="store_true", help="Fail if an argument set has no hardware CSV")
    p.add_argument("--jobs", type=int, default=3, help="Argument sets packed and uploaded in parallel")
    p.add_argument(
        "--readme",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "trace-release-README.md"),
        help="Template uploaded as the release's README.md, with a table of the sets appended",
    )
    p.add_argument("only", nargs="*", help="Optional <workload>/<args> subset")
    o = p.parse_args()

    root = os.path.join(o.src, o.trace_root)
    sets = sorted(
        tuple(os.path.relpath(k, root).split(os.sep)[:2])
        for k in glob.glob(os.path.join(root, "*", "*", "traces", "kernelslist.g"))
    )
    if o.only:
        sets = [s for s in sets if f"{s[0]}/{s[1]}" in o.only]
        missing = set(o.only) - {f"{a}/{b}" for a, b in sets}
        assert not missing, f"not found under {root}: {sorted(missing)}"
    assert sets, f"no <workload>/<args>/traces/kernelslist.g under {root}"
    pub = Publisher(o)
    print(f"{len(sets)} argument sets -> {o.repo} release {o.tag}", flush=True)
    with ThreadPoolExecutor(o.jobs) as ex:
        list(ex.map(lambda s: pub.one(*s), sets))
    entries = pub.write_manifest()
    print(f"manifest.json: {len(entries)} argument sets, {sum(e['size'] for e in entries) / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
