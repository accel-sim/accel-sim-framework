#!/usr/bin/env python3
"""Download pre-collected Accel-Sim SASS traces from a GitHub release.

Each trace set (one GPU / workload / argument set) is a tar split into parts
smaller than 2 GB. The release's manifest.json lists every part with its
SHA-256; this script downloads the selected parts, verifies them, and extracts
the tar under <download_dir> (default ./hw_run) in the standard Accel-Sim layout:
  <GPU>/traces/device-0/<cuda>/<workload>/<args>/traces/   run_simulations.py -T
  <GPU>/device-0/<cuda>/<workload>/<args>/ncu.csv.*         plot-correlation.py -H <GPU>

Private repositories need a token: set GITHUB_TOKEN, or log in with `gh`.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request

DEFAULT_REPO = "accel-sim/accel-sim-framework"
# Traces ship as assets of the code release they were validated with.
DEFAULT_TAG = "v2.0.1"
API = "https://api.github.com"
CHUNK = 1 << 22
EXTRACT_KW = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
this_directory = os.path.dirname(os.path.realpath(__file__))


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000.0


def get_token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    if shutil.which("gh"):
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_url(url, token=None, accept=None):
    """Open url; with a token, follow the asset redirect without forwarding Authorization."""
    headers = {"User-Agent": "accel-sim-get-traces"}
    if accept:
        headers["Accept"] = accept
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = urllib.request.build_opener(NoRedirect)
    try:
        return opener.open(urllib.request.Request(url, headers=headers))
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            loc = e.headers["Location"]
            return urllib.request.urlopen(urllib.request.Request(loc, headers={"User-Agent": headers["User-Agent"]}))
        raise


class Release:
    def __init__(self, repo, tag, token):
        self.repo, self.tag, self.token = repo, tag, token
        try:
            with open_url(f"{API}/repos/{repo}/releases/tags/{tag}", token, "application/vnd.github+json") as r:
                rel = json.load(r)
        except urllib.error.HTTPError as e:
            hint = "" if token else " (private repositories need GITHUB_TOKEN or `gh auth login`)"
            sys.exit(f"Cannot read release {tag} of {repo}: HTTP {e.code}{hint}")
        self.assets = {a["name"]: a for a in rel["assets"]}

    def open_asset(self, name):
        if name not in self.assets:
            sys.exit(f"Asset {name} is not in release {self.tag} of {self.repo}")
        a = self.assets[name]
        if self.token:
            return open_url(a["url"], self.token, "application/octet-stream")
        return open_url(a["browser_download_url"])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def download_part(release, part, dest):
    """Download one part to dest, skipping it if a verified copy is already there."""
    if os.path.exists(dest) and os.path.getsize(dest) == part["size"] and sha256_file(dest) == part["sha256"]:
        print(f"  {part['name']}: already downloaded")
        return
    tmp = dest + ".tmp"
    h, done = hashlib.sha256(), 0
    with release.open_asset(part["name"]) as r, open(tmp, "wb") as f:
        for b in iter(lambda: r.read(CHUNK), b""):
            f.write(b)
            h.update(b)
            done += len(b)
            print(f"\r  {part['name']}: {human(done)} / {human(part['size'])}", end="", flush=True)
    print()
    if done != part["size"] or h.hexdigest() != part["sha256"]:
        os.remove(tmp)
        sys.exit(f"{part['name']}: size/SHA-256 mismatch (got {done} bytes, expected {part['size']})")
    os.replace(tmp, dest)


class Concat:
    """Read a list of files as one stream, hashing everything read."""

    def __init__(self, paths):
        self.paths, self.f, self.h = list(paths), None, hashlib.sha256()

    def read(self, n=-1):
        out = b""
        while n < 0 or len(out) < n:
            if self.f is None:
                if not self.paths:
                    break
                self.f = open(self.paths.pop(0), "rb")
            b = self.f.read(-1 if n < 0 else n - len(out))
            if not b:
                self.f.close()
                self.f = None
                continue
            out += b
        self.h.update(out)
        return out


def set_name(e):
    return f"{e['gpu']}/{e['workload']}/{e['args']}"


def extract(entry, part_paths, out_dir):
    """Extract one set; every member must be a file or directory inside its trace or hw directory."""
    allowed = [d for d in (entry["trace_dir"], entry["hw_dir"]) if d]
    stream = Concat(part_paths)
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        root = os.path.realpath(out_dir)
        for m in tar:
            target = os.path.realpath(os.path.join(out_dir, m.name))
            if not target.startswith(root + os.sep) or not (m.isfile() or m.isdir()):
                sys.exit(f"Refusing unexpected tar member {m.name}")
            if not any(m.name == d or m.name.startswith(d + "/") for d in allowed):
                sys.exit(f"Tar member {m.name} is outside {allowed}")
            tar.extract(m, out_dir, **EXTRACT_KW)
    while stream.read(CHUNK):  # hash trailing padding so the whole-tar checksum covers every byte
        pass
    if stream.h.hexdigest() != entry["sha256"]:
        sys.exit(f"{set_name(entry)}: whole-tar SHA-256 mismatch after extraction")
    listed = [l for l in open(os.path.join(out_dir, entry["trace_dir"], "kernelslist.g")) if l.startswith("kernel-")]
    if len(listed) != entry["kernels"]:
        sys.exit(f"{set_name(entry)}: kernelslist.g has {len(listed)} kernels, manifest says {entry['kernels']}")


def select(manifest, selection):
    """selection: comma list of GPU[/workload[/args]], each level may be 'all'."""
    chosen = []
    for item in selection.split(","):
        want = [x for x in item.strip().split("/") if x]
        if not want:
            continue
        hits = [e for e in manifest["traces"]
                if all(w in ("all", v) for w, v in zip(want, (e["gpu"], e["workload"], e["args"])))]
        if not hits:
            raise ValueError(f"nothing matches '{item}'")
        chosen += [e for e in hits if e not in chosen]
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-a", "--apps", help="Comma separated GPU[/workload[/args]] list, 'all' allowed at any level. "
                   "Example: -a H200/inference_bert.sh/32_256,H200/inference_sdxl.sh")
    p.add_argument("-d", "--download_dir", default=os.path.join(this_directory, "hw_run"),
                   help="Extract into <download_dir>/<GPU>/ (default: ./hw_run)")
    p.add_argument("-l", "--list", action="store_true", help="List the available traces and exit")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repository (default: {DEFAULT_REPO})")
    p.add_argument("--tag", default=DEFAULT_TAG, help=f"Release tag (default: {DEFAULT_TAG})")
    p.add_argument("--keep_parts", action="store_true", help="Keep the downloaded tar parts after extraction")
    o = p.parse_args()

    release = Release(o.repo, o.tag, get_token())
    with release.open_asset("manifest.json") as r:
        manifest = json.load(r)
    assert manifest.get("format") == 2, f"unsupported manifest format {manifest.get('format')}, expected 2"

    print(f"Available traces in {o.repo} release {o.tag}:")
    for e in sorted(manifest["traces"], key=lambda e: (e["gpu"], e["workload"], e["args"])):
        print(f"  {e['gpu']}/{e['workload']}/{e['args']}: {e['kernels']} kernels, {human(e['size'])}")
    total = sum(e["size"] for e in manifest["traces"])
    print(f"  total: {len(manifest['traces'])} trace sets, {human(total)}")
    if o.list:
        return

    while True:
        selection = o.apps or input("\nWhat do you want to download? GPU[/workload[/args]],... (default all): ") or "all"
        try:
            chosen = select(manifest, selection)
            break
        except ValueError as e:
            print(f"Invalid input: {e}")
            if o.apps:
                sys.exit(1)

    out_dir = os.path.abspath(o.download_dir)
    parts_dir = os.path.join(out_dir, ".trace-parts")
    os.makedirs(parts_dir, exist_ok=True)
    print(f"\nDownloading {len(chosen)} trace sets, {human(sum(e['size'] for e in chosen))}, into {out_dir}")
    for e in chosen:
        print(f"\n{e['gpu']}/{e['workload']}/{e['args']} ({len(e['parts'])} parts)")
        paths = []
        for part in e["parts"]:
            dest = os.path.join(parts_dir, part["name"])
            download_part(release, part, dest)
            paths.append(dest)
        extract(e, paths, out_dir)
        if not o.keep_parts:
            for path in paths:
                os.remove(path)
        print(f"  extracted to {os.path.join(out_dir, e['trace_dir'])}")
    if not o.keep_parts:
        try:
            os.rmdir(parts_dir)
        except OSError:
            pass
    print("\nDone.")
    for t, h in sorted({(e["trace_root"], e["hw_root"]) for e in chosen}):
        gpu_dir = os.path.join(out_dir, t.split("/")[0])
        print(f"  run_simulations.py -T {os.path.join(out_dir, t)}")
        print(f"  plot-correlation.py -H {gpu_dir}")


if __name__ == "__main__":
    main()
