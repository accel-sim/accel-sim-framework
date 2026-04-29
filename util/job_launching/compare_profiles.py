#!/usr/bin/env python3

# Compare profiling results from one or two simulation runs.
# Supports perf record (CPU profiling) and heaptrack (memory profiling) data.
#
# Usage:
#   # Single run - just plot
#   ./compare_profiles.py -A ./sim_run_12.8 -o ./results
#
#   # Compare two runs
#   ./compare_profiles.py -A ./sim_run_baseline -B ./sim_run_other \
#       --baseline-name "h100-test" --other-name "memory-opt" -o ./comparison

from optparse import OptionParser
import os
import re
import sys
import subprocess
import glob
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# Cache for the extracted heaptrack_print path (set once, used by all threads)
_heaptrack_print_path = None


def resolve_heaptrack_print(heaptrack_bin):
    """Find or extract heaptrack_print once, return its path.
    Thread-safe: called once from main thread before spawning workers."""
    global _heaptrack_print_path
    if _heaptrack_print_path is not None:
        return _heaptrack_print_path

    # 1. heaptrack_print in PATH
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(p, "heaptrack_print")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            _heaptrack_print_path = candidate
            return _heaptrack_print_path

    # 2. heaptrack_print beside the heaptrack binary
    if heaptrack_bin and heaptrack_bin != "heaptrack":
        beside = os.path.join(os.path.dirname(heaptrack_bin), "heaptrack_print")
        if os.path.isfile(beside):
            _heaptrack_print_path = beside
            return _heaptrack_print_path

        # 3. Extract from AppImage (once, to a temp directory)
        if os.path.isfile(heaptrack_bin):
            extract_dir = tempfile.mkdtemp(prefix="heaptrack_extract_")
            try:
                subprocess.run(
                    [heaptrack_bin, "--appimage-extract"],
                    cwd=extract_dir, capture_output=True, timeout=60,
                )
                extracted = os.path.join(
                    extract_dir, "squashfs-root", "usr", "bin", "heaptrack_print"
                )
                if os.path.isfile(extracted):
                    _heaptrack_print_path = extracted
                    print("Extracted heaptrack_print to {}".format(extracted),
                          file=sys.stderr)
                    return _heaptrack_print_path
            except (subprocess.TimeoutExpired, OSError):
                pass

    _heaptrack_print_path = ""  # empty string = not found
    return _heaptrack_print_path
sys.path.insert(0, this_directory)
import common


def parse_options():
    parser = OptionParser(
        usage="Compare profiling results between simulation runs.\n"
        "  %prog --baseline <sim_run_dir> [options]\n"
        "  %prog --baseline <baseline_dir> --other <other_dir> [options]"
    )
    parser.add_option(
        "-A",
        "--baseline",
        dest="baseline_dir",
        help="Path to the first (or only) sim_run directory.",
    )
    parser.add_option(
        "-B",
        "--other",
        dest="other_dir",
        default=None,
        help="Path to the second sim_run directory (enables comparison mode).",
    )
    parser.add_option(
        "--baseline-name",
        dest="baseline_name",
        default="baseline",
        help="Display name for --baseline run (default: 'baseline').",
    )
    parser.add_option(
        "--other-name",
        dest="other_name",
        default="other",
        help="Display name for --other run (default: 'other').",
    )
    parser.add_option(
        "-o",
        "--output",
        dest="output_dir",
        default="profile_comparison",
        help="Directory to write report and plots (default: profile_comparison).",
    )
    parser.add_option(
        "--heaptrack-bin",
        dest="heaptrack_bin",
        default="heaptrack",
        help="Path to heaptrack binary or AppImage (default: 'heaptrack').",
    )
    parser.add_option(
        "-j",
        "--jobs",
        dest="jobs",
        type="int",
        default=os.cpu_count(),
        help="Number of parallel parsing jobs (default: CPU count).",
    )
    (options, args) = parser.parse_args()
    if not options.baseline_dir:
        parser.error("-A (baseline directory) is required.")
    return options


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_perf_report(perf_data_path):
    """Run perf report on a perf.data file and extract top functions."""
    result = {"functions": []}
    if not os.path.isfile(perf_data_path):
        return result
    try:
        proc = subprocess.run(
            [
                "perf",
                "report",
                "--stdio",
                "-n",
                "--no-children",
                "-i",
                perf_data_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in proc.stdout.split("\n"):
            # Match: "  9.18%  6193  accel-sim.out  accel-sim.out  [.] function_name"
            m = re.match(
                r"\s+([\d.]+)%\s+(\d+)\s+\S+\s+\S+\s+\[.\]\s+(.*)", line
            )
            if m:
                result["functions"].append(
                    {
                        "overhead_pct": float(m.group(1)),
                        "samples": int(m.group(2)),
                        "symbol": m.group(3).strip(),
                    }
                )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return result


def _parse_heap_size(size_str):
    """Parse heap size string like '617.24M' to float MB."""
    m = re.match(r"([\d.]+)\s*([KMG]?)", str(size_str))
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "G":
        val *= 1024
    elif unit == "K":
        val /= 1024
    return val


def parse_heaptrack(zst_path, heaptrack_bin):
    """Run heaptrack_print on a .zst file and parse summary stats."""
    result = {}
    # heaptrack appends .zst if not present
    if not os.path.isfile(zst_path):
        zst_path = zst_path + ".zst"
    if not os.path.isfile(zst_path):
        return result
    # Use the pre-resolved heaptrack_print path (extracted once, thread-safe)
    ht_print = _heaptrack_print_path
    if not ht_print:
        return result
    try:
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        env.pop("QT_QPA_PLATFORM", None)
        proc = subprocess.run(
            [ht_print, "-f", zst_path],
            capture_output=True, text=True, timeout=600, env=env,
        )
        output = proc.stdout + proc.stderr
        if "peak heap" not in output:
            return result

        lines = output.split("\n")
        for line in lines:
            m = re.match(r"peak heap memory consumption:\s+(.*)", line)
            if m:
                result["peak_heap"] = m.group(1).strip()
            m = re.match(r"total memory leaked:\s+(.*)", line)
            if m:
                result["total_leaked"] = m.group(1).strip()
            m = re.match(r"calls to allocation functions:\s+(\d+)", line)
            if m:
                result["total_allocations"] = int(m.group(1))
            m = re.match(r"temporary memory allocations:\s+(\d+)", line)
            if m:
                result["temp_allocations"] = int(m.group(1))

        # Parse top allocation sites:
        # Format: "<N> calls to allocation functions with <size> peak consumption from\n<function>"
        top_sites = []
        for i, line in enumerate(lines):
            m = re.match(
                r"(\d+) calls to allocation functions with ([\d.]+\s*[KMG]?B?) peak consumption from",
                line,
            )
            if m and i + 1 < len(lines):
                func_name = lines[i + 1].strip()
                if func_name and not func_name.startswith("in "):
                    peak_str = m.group(2)
                    peak_mb = _parse_heap_size(peak_str)
                    top_sites.append({
                        "calls": int(m.group(1)),
                        "peak_mb": peak_mb,
                        "function": func_name,
                    })
        # Sort by peak consumption descending, take top 15
        top_sites.sort(key=lambda s: s["peak_mb"], reverse=True)
        result["top_alloc_sites"] = top_sites[:15]
    except subprocess.TimeoutExpired:
        pass
    return result


def parse_sim_stats(output_dir):
    """Extract app-level and per-kernel stats from simulation output.

    Returns dict with:
      "app": {stat_name: value}  -- final aggregate stats for the whole app
      "kernels": [{"name": kname, stat_name: value}, ...]  -- per-kernel stats
    """
    result = {"app": {}, "kernels": []}
    ofiles = glob.glob(os.path.join(output_dir, "*.o*"))
    ofiles = [f for f in ofiles if re.search(r"\.o\d+$", f)]
    if not ofiles:
        return result
    # Prefer the unprofiled run output for accurate simulation stats.
    # Output filenames: normal = "name.o<jobid>", perf = "name.perf.o<jobid>",
    # mem = "name.mem.o<jobid>".  Select files that do NOT have .perf. or .mem.
    # immediately before the .o<jobid> suffix.
    normal_ofiles = [f for f in ofiles
                     if not re.search(r"\.(perf|mem)\.o\d+$", f)]
    if normal_ofiles:
        outfile = max(normal_ofiles, key=os.path.getmtime)
    else:
        outfile = max(ofiles, key=os.path.getmtime)
        print("WARNING: No unprofiled output file found in {0}, "
              "using profiled output {1} (stats may include profiling "
              "overhead)".format(output_dir, os.path.basename(outfile)),
              file=sys.stderr)

    # Aggregate stats (cumulative — need differencing for per-kernel)
    agg_patterns = {
        "gpu_tot_sim_cycle": re.compile(r"gpu_tot_sim_cycle\s*=\s*([\d]+)"),
        "gpu_tot_sim_insn": re.compile(r"gpu_tot_sim_insn\s*=\s*([\d]+)"),
        "gpgpu_simulation_time": re.compile(r"gpgpu_simulation_time\s*=.*\((\d+) sec\)"),
    }
    # Absolute stats (per-kernel, not cumulative)
    abs_patterns = {
        "gpu_ipc": re.compile(r"gpu_ipc\s*=\s*([\d.]+)"),
    }
    # Rate stats (snapshot, only meaningful for final kernel)
    rate_patterns = {
        "gpu_tot_ipc": re.compile(r"gpu_tot_ipc\s*=\s*([\d.]+)"),
        "gpgpu_simulation_rate_ips": re.compile(
            r"gpgpu_simulation_rate\s*=\s*([\d.]+)\s*\(inst/sec\)"
        ),
    }
    kernel_name_pat = re.compile(r"kernel_name\s+=\s+(.*)")

    try:
        with open(outfile) as f:
            current_kernel = ""
            running_kcount = {}
            raw_last = {}  # last raw cumulative values for differencing
            kernel_data = {}  # kname -> {stat: value}

            for line in f:
                km = kernel_name_pat.match(line)
                if km:
                    current_kernel = km.group(1).strip()
                    if current_kernel not in running_kcount:
                        running_kcount[current_kernel] = 0
                    else:
                        running_kcount[current_kernel] += 1
                    current_kernel += "--" + str(running_kcount[current_kernel])
                    kernel_data[current_kernel] = {"name": current_kernel}
                    continue

                if not current_kernel:
                    continue

                # Aggregate stats: difference from last kernel
                for name, pat in agg_patterns.items():
                    m = pat.search(line)
                    if m:
                        val = float(m.group(1))
                        prev = raw_last.get(name, 0.0)
                        raw_last[name] = val
                        kernel_data[current_kernel][name] = val - prev

                # Absolute stats: store directly
                for name, pat in abs_patterns.items():
                    m = pat.search(line)
                    if m:
                        kernel_data[current_kernel][name] = float(m.group(1))

                # Rate stats: store directly (overwritten each kernel)
                for name, pat in rate_patterns.items():
                    m = pat.search(line)
                    if m:
                        kernel_data[current_kernel][name] = float(m.group(1))

        # Build kernel list in order
        result["kernels"] = list(kernel_data.values())

        # App-level: final cumulative values (aggregated across all kernels)
        # raw_last holds the final cumulative counter values
        if raw_last:
            result["app"]["gpu_tot_sim_cycle"] = str(int(raw_last.get("gpu_tot_sim_cycle", 0)))
            result["app"]["gpu_tot_sim_insn"] = str(int(raw_last.get("gpu_tot_sim_insn", 0)))
            tot_time = raw_last.get("gpgpu_simulation_time", 0)
            tot_insn = raw_last.get("gpu_tot_sim_insn", 0)
            tot_cycle = raw_last.get("gpu_tot_sim_cycle", 0)
            result["app"]["gpgpu_simulation_time"] = str(int(tot_time))
            # Aggregate simulation rate = total insn / total wall time
            if tot_time > 0:
                result["app"]["gpgpu_simulation_rate_ips"] = str(int(tot_insn / tot_time))
            # Aggregate IPC = total insn / total cycles
            if tot_cycle > 0:
                result["app"]["gpu_tot_ipc"] = str(tot_insn / tot_cycle)
    except (IOError, OSError):
        pass
    return result


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_benchmarks(run_dir):
    """Find all benchmark/args/config directories in a sim_run directory."""
    results = []
    if not os.path.isdir(run_dir):
        return results
    for bench_dir in sorted(os.listdir(run_dir)):
        bench_path = os.path.join(run_dir, bench_dir)
        if not os.path.isdir(bench_path) or bench_dir == "gpgpu-sim-builds":
            continue
        for args_dir in sorted(os.listdir(bench_path)):
            args_path = os.path.join(bench_path, args_dir)
            if not os.path.isdir(args_path):
                continue
            for config_dir in sorted(os.listdir(args_path)):
                config_path = os.path.join(args_path, config_dir)
                if not os.path.isdir(config_path):
                    continue
                key = os.path.join(bench_dir, args_dir, config_dir)
                results.append((key, config_path))
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def compute_delta_pct(base, opt):
    try:
        b, o = float(base), float(opt)
        if b == 0:
            return "N/A"
        return "{:+.1f}%".format((o - b) / b * 100)
    except (ValueError, TypeError):
        return "N/A"


def generate_text_report(data, options):
    comparison = options.other_dir is not None
    sep = "-" * 90

    # Simulation stats
    print("\n" + sep)
    print("SIMULATOR STATS")
    if comparison:
        fmt = "{:<55} {:>12} {:>12} {:>8}"
        print(fmt.format("Benchmark", options.baseline_name, options.other_name, "Delta"))
    else:
        fmt = "{:<55} {:>12}"
        print(fmt.format("Benchmark", options.baseline_name))
    print(sep)
    for entry in data:
        label = entry["key"][-55:]
        base_rate = entry["base_sim"]["app"].get("gpgpu_simulation_rate_ips", "N/A")
        if comparison:
            opt_rate = entry["opt_sim"]["app"].get("gpgpu_simulation_rate_ips", "N/A")
            delta = compute_delta_pct(base_rate, opt_rate)
            print(fmt.format(label, str(base_rate), str(opt_rate), delta))
            base_cyc = entry["base_sim"]["app"].get("gpu_tot_sim_cycle", "N/A")
            opt_cyc = entry["opt_sim"]["app"].get("gpu_tot_sim_cycle", "N/A")
            match = "MATCH" if base_cyc == opt_cyc else "MISMATCH"
            print("  Correctness: {}".format(match))
        else:
            print(fmt.format(label, str(base_rate)))

    # Perf report top functions
    print("\n" + sep)
    print("TOP CPU FUNCTIONS (perf record)")
    print(sep)
    for entry in data:
        funcs = entry["base_perf"].get("functions", [])
        if funcs:
            print("\n  {} ({}):".format(entry["key"], options.baseline_name))
            for f in funcs[:10]:
                print(
                    "    {:.1f}%  {:>8} samples  {}".format(
                        f["overhead_pct"], f["samples"], f["symbol"]
                    )
                )
        if comparison:
            opt_funcs = entry.get("opt_perf", {}).get("functions", [])
            if opt_funcs:
                print("\n  {} ({}):".format(entry["key"], options.other_name))
                for f in opt_funcs[:10]:
                    print(
                        "    {:.1f}%  {:>8} samples  {}".format(
                            f["overhead_pct"], f["samples"], f["symbol"]
                        )
                    )

    # Heaptrack summary
    print("\n" + sep)
    print("HEAP MEMORY (heaptrack)")
    print(sep)
    for entry in data:
        ht = entry.get("base_heaptrack", {})
        if not ht:
            continue
        print("\n  {} ({}):".format(entry["key"], options.baseline_name))
        for k, v in ht.items():
            print("    {}: {}".format(k, v))
        if comparison:
            oht = entry.get("opt_heaptrack", {})
            if oht:
                print("  {} ({}):".format(entry["key"], options.other_name))
                for k, v in oht.items():
                    print("    {}: {}".format(k, v))


def generate_plots(data, options):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed, skipping HTML chart generation.")
        return

    os.makedirs(options.output_dir, exist_ok=True)
    comparison = options.other_dir is not None

    # Label helpers: indexed app name for x-axis, full key for hover
    def _app_name(key):
        return key.split("/")[0]

    def _full_label(key):
        return "/".join(key.split("/")[:2])

    # Use "appname #N" so different args for the same app get separate bars
    app_counter = {}
    app_labels = []
    for e in data:
        name = _app_name(e["key"])
        idx = app_counter.get(name, 0)
        app_counter[name] = idx + 1
        app_labels.append("{} #{}".format(name, idx) if app_counter[name] > 1 or
                          sum(1 for e2 in data if _app_name(e2["key"]) == name) > 1
                          else name)
    hover_labels = [_full_label(e["key"]) for e in data]

    # Collect sections: scatter plots, bar charts, per-benchmark hotspots
    scatter_figs = []
    bar_figs = []
    hotspot_figs = []

    # Colors
    C_BASE = "#0F8C79"
    C_OTHER = "#BD2D28"
    C_SPEED = "#E3BA22"

    # ---- Collect app-level numeric data ----

    base_rates = []
    opt_rates = []
    for e in data:
        try:
            base_rates.append(float(e["base_sim"]["app"].get("gpgpu_simulation_rate_ips", 0)))
        except (ValueError, TypeError):
            base_rates.append(0)
        if comparison:
            try:
                opt_rates.append(float(e["opt_sim"]["app"].get("gpgpu_simulation_rate_ips", 0)))
            except (ValueError, TypeError):
                opt_rates.append(0)

    base_heaps = []
    opt_heaps = []
    base_allocs = []
    opt_allocs = []
    ht_app_labels = []
    ht_hover_labels = []
    for i, e in enumerate(data):
        bht = e.get("base_heaptrack", {})
        if not bht:
            continue
        ht_app_labels.append(app_labels[i])
        ht_hover_labels.append(hover_labels[i])
        base_heaps.append(_parse_heap_size(bht.get("peak_heap", "0")))
        base_allocs.append(bht.get("total_allocations", 0))
        if comparison:
            oht = e.get("opt_heaptrack", {})
            opt_heaps.append(_parse_heap_size(oht.get("peak_heap", "0")))
            opt_allocs.append(oht.get("total_allocations", 0))

    # ---- Collect per-kernel simulation rate data ----

    kernel_labels_base = []
    kernel_rates_base = []
    kernel_labels_opt = []
    kernel_rates_opt = []
    for e in data:
        app = _app_name(e["key"])
        for k in e["base_sim"].get("kernels", []):
            t = k.get("gpgpu_simulation_time", 0)
            insn = k.get("gpu_tot_sim_insn", 0)
            rate = insn / t if t > 0 else 0
            kernel_labels_base.append("{}/{}".format(app, k["name"]))
            kernel_rates_base.append(rate)
        if comparison:
            for k in e["opt_sim"].get("kernels", []):
                t = k.get("gpgpu_simulation_time", 0)
                insn = k.get("gpu_tot_sim_insn", 0)
                rate = insn / t if t > 0 else 0
                kernel_labels_opt.append("{}/{}".format(app, k["name"]))
                kernel_rates_opt.append(rate)

    # ---- SECTION 1: Correlation scatter plots (comparison only, first) ----

    if comparison:

        def _add_correl_scatter(x_vals, y_vals, point_labels, title, axis_label,
                                log_scale=False):
            if not x_vals or not any(v > 0 for v in x_vals):
                return
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode="markers",
                    marker=dict(size=10, color=C_BASE),
                    hovertemplate="%{customdata}<br>"
                    + options.baseline_name + ": %{x}<br>"
                    + options.other_name + ": %{y}<extra></extra>",
                    customdata=point_labels,
                )
            )
            all_vals = [v for v in x_vals + y_vals if v > 0]
            if all_vals:
                lo = min(all_vals) * 0.8
                hi = max(all_vals) * 1.2
                fig.add_trace(
                    go.Scatter(
                        x=[lo, hi], y=[lo, hi], mode="lines",
                        line=dict(dash="dash", color="red"),
                        name="y = x", showlegend=True,
                    )
                )
            axis_type = "log" if log_scale else "linear"
            fig.update_layout(
                title=title,
                xaxis_title="{} ({})".format(axis_label, options.baseline_name),
                yaxis_title="{} ({})".format(axis_label, options.other_name),
                xaxis_type=axis_type,
                yaxis_type=axis_type,
                height=700, width=800,
            )
            scatter_figs.append(fig)

        # Simulation cycle correlation (correctness check — should be on y=x)
        base_cycles = []
        opt_cycles = []
        for e in data:
            try:
                base_cycles.append(float(e["base_sim"]["app"].get("gpu_tot_sim_cycle", 0)))
            except (ValueError, TypeError):
                base_cycles.append(0)
            try:
                opt_cycles.append(float(e["opt_sim"]["app"].get("gpu_tot_sim_cycle", 0)))
            except (ValueError, TypeError):
                opt_cycles.append(0)
        if base_cycles and opt_cycles and any(v > 0 for v in base_cycles):
            _add_correl_scatter(base_cycles, opt_cycles, hover_labels,
                                "[App] Simulation Cycle Correlation<br>"
                                "<sub>Points off y=x indicate correctness differences</sub>",
                                "cycles", log_scale=True)

        if base_rates and opt_rates:
            _add_correl_scatter(base_rates, opt_rates, hover_labels,
                                "[App] Simulation Rate Correlation (⬆️ better)<br>"
                                "<sub>Aggregate: total_insn / total_wall_time</sub>",
                                "inst/sec")
        if ht_hover_labels and base_heaps and opt_heaps:
            _add_correl_scatter(base_heaps, opt_heaps, ht_hover_labels,
                                "[App] Peak Heap Memory Correlation (⬇️ better)", "MB")
        if ht_hover_labels and base_allocs and opt_allocs:
            _add_correl_scatter([float(a) for a in base_allocs],
                                [float(a) for a in opt_allocs], ht_hover_labels,
                                "[App] Allocation Count Correlation (⬇️ better)", "calls")

    # ---- SECTION 2: App-level bar charts ----

    # [App] Simulation rate
    bars = [go.Bar(name=options.baseline_name, x=app_labels, y=base_rates,
                   marker_color=C_BASE, hovertext=hover_labels)]
    if comparison:
        bars.append(go.Bar(name=options.other_name, x=app_labels, y=opt_rates,
                           marker_color=C_OTHER, hovertext=hover_labels))
    fig = go.Figure(data=bars)
    fig.update_layout(title="[App] Simulation Rate (inst/sec)<br>"
                      "<sub>Aggregate: total_insn / total_wall_time across all kernel launches</sub>",
                      barmode="group", yaxis_title="inst/sec", height=500)
    bar_figs.append(fig)

    # [App] Speedup
    if comparison and base_rates and opt_rates:
        speedups = [o / b if b > 0 else 0 for b, o in zip(base_rates, opt_rates)]
        # Compute geometric mean of non-zero speedups
        valid_speedups = [s for s in speedups if s > 0]
        if valid_speedups:
            import math
            geomean = math.exp(sum(math.log(s) for s in valid_speedups) / len(valid_speedups))
        else:
            geomean = 0
        all_labels = app_labels + ["GeoMean"]
        all_speedups = speedups + [geomean]
        all_hovers = hover_labels + ["Geometric Mean ({:.3f}x)".format(geomean)]
        colors = [C_SPEED] * len(speedups) + ["#333333"]
        fig = go.Figure(data=[go.Bar(x=all_labels, y=all_speedups,
                                     marker_color=colors, hovertext=all_hovers)])
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                      annotation_text="No change (1.0x)")
        fig.update_layout(
            title="[App] Simulation Rate Speedup ({} / {})<br>"
            "<sub>GeoMean = {:.3f}x</sub>".format(
                options.other_name, options.baseline_name, geomean),
            yaxis_title="Speedup (x)", height=500)
        bar_figs.append(fig)

    # [Kernel] Per-kernel simulation rate correlation scatter
    if comparison and kernel_rates_base and kernel_rates_opt:
        # Match kernels by index (same order from same benchmark suite)
        n = min(len(kernel_rates_base), len(kernel_rates_opt))
        if n > 0:
            _add_correl_scatter(
                kernel_rates_base[:n], kernel_rates_opt[:n],
                kernel_labels_base[:n],
                "[Kernel] Per-Kernel Simulation Rate Correlation (⬆️ better) <br>"
                "<sub>Each point = one kernel launch instance; rate = insn_delta / time_delta</sub>",
                "inst/sec",
            )

    # [App] Peak heap
    if ht_app_labels:
        bars = [go.Bar(name=options.baseline_name, x=ht_app_labels, y=base_heaps,
                       marker_color=C_BASE, hovertext=ht_hover_labels)]
        if comparison and opt_heaps:
            bars.append(go.Bar(name=options.other_name, x=ht_app_labels, y=opt_heaps,
                               marker_color=C_OTHER, hovertext=ht_hover_labels))
        fig = go.Figure(data=bars)
        fig.update_layout(title="[App] Peak Heap Memory (MB)", barmode="group",
                          yaxis_title="MB", height=500)
        bar_figs.append(fig)

        # [App] Allocation count
        bars = [go.Bar(name=options.baseline_name, x=ht_app_labels, y=base_allocs,
                       marker_color=C_BASE, hovertext=ht_hover_labels)]
        if comparison and opt_allocs:
            bars.append(go.Bar(name=options.other_name, x=ht_app_labels, y=opt_allocs,
                               marker_color=C_OTHER, hovertext=ht_hover_labels))
        fig = go.Figure(data=bars)
        fig.update_layout(title="[App] Heap Allocation Calls", barmode="group",
                          yaxis_title="Count", height=500)
        bar_figs.append(fig)

    # ---- SECTION 3: Per-benchmark hotspot charts (CPU + memory) ----

    for e in data:
        bench_label = e["key"]

        # CPU hotspots — side-by-side overhead % + absolute samples
        base_funcs = e.get("base_perf", {}).get("functions", [])[:15]
        opt_funcs = e.get("opt_perf", {}).get("functions", [])[:15] if comparison else []

        if base_funcs:
            # Merge function lists (union of both, ordered by max overhead)
            func_map = {}
            for f in base_funcs:
                func_map[f["symbol"]] = {"base_pct": f["overhead_pct"],
                                          "base_samples": f["samples"],
                                          "opt_pct": 0, "opt_samples": 0}
            for f in opt_funcs:
                if f["symbol"] in func_map:
                    func_map[f["symbol"]]["opt_pct"] = f["overhead_pct"]
                    func_map[f["symbol"]]["opt_samples"] = f["samples"]
                else:
                    func_map[f["symbol"]] = {"base_pct": 0, "base_samples": 0,
                                              "opt_pct": f["overhead_pct"],
                                              "opt_samples": f["samples"]}
            # Sort by max overhead across both runs
            sorted_funcs = sorted(func_map.items(),
                                  key=lambda x: max(x[1]["base_pct"], x[1]["opt_pct"]))
            syms = [k[:60] + "..." if len(k) > 60 else k for k, _ in sorted_funcs]
            full_syms = [k for k, _ in sorted_funcs]

            # Overhead % side-by-side (optimized first so baseline renders on top)
            traces = []
            if comparison and opt_funcs:
                traces.append(go.Bar(
                    name=options.other_name, orientation="h",
                    x=[func_map[k]["opt_pct"] for k, _ in sorted_funcs],
                    y=syms, marker_color=C_OTHER,
                    hovertemplate="%{customdata}<br>Overhead: %{x:.1f}%<extra></extra>",
                    customdata=full_syms,
                ))
            traces.append(go.Bar(
                name=options.baseline_name, orientation="h",
                x=[func_map[k]["base_pct"] for k, _ in sorted_funcs],
                y=syms, marker_color=C_BASE,
                hovertemplate="%{customdata}<br>Overhead: %{x:.1f}%<extra></extra>",
                customdata=full_syms,
            ))
            fig = go.Figure(data=traces)
            fig.update_layout(
                title="CPU Hotspots (Overhead %): {}".format(bench_label),
                xaxis_title="Overhead %", barmode="group",
                yaxis=dict(automargin=True, tickfont=dict(size=11)),
                height=max(450, len(sorted_funcs) * 30 + 120),
                margin=dict(l=0), yaxis_automargin=True,
            )
            hotspot_figs.append(fig)

            # Absolute samples side-by-side
            if comparison and opt_funcs:
                traces = [
                    go.Bar(
                        name=options.other_name, orientation="h",
                        x=[func_map[k]["opt_samples"] for k, _ in sorted_funcs],
                        y=syms, marker_color=C_OTHER,
                        hovertemplate="%{customdata}<br>Samples: %{x}<extra></extra>",
                        customdata=full_syms,
                    ),
                    go.Bar(
                        name=options.baseline_name, orientation="h",
                        x=[func_map[k]["base_samples"] for k, _ in sorted_funcs],
                        y=syms, marker_color=C_BASE,
                        hovertemplate="%{customdata}<br>Samples: %{x}<extra></extra>",
                        customdata=full_syms,
                    ),
                ]
                fig = go.Figure(data=traces)
                fig.update_layout(
                    title="CPU Hotspots (Absolute Samples): {}".format(bench_label),
                    xaxis_title="Samples", barmode="group",
                    yaxis=dict(automargin=True, tickfont=dict(size=11)),
                    height=max(450, len(sorted_funcs) * 30 + 120),
                    margin=dict(l=0), yaxis_automargin=True,
                )
                hotspot_figs.append(fig)

        # Memory hotspots — side-by-side peak MB + absolute calls
        base_sites = e.get("base_heaptrack", {}).get("top_alloc_sites", [])[:15]
        opt_sites = e.get("opt_heaptrack", {}).get("top_alloc_sites", [])[:15] if comparison else []

        if base_sites:
            # Merge allocation sites
            site_map = {}
            for s in base_sites:
                site_map[s["function"]] = {"base_mb": s["peak_mb"], "base_calls": s["calls"],
                                            "opt_mb": 0, "opt_calls": 0}
            for s in opt_sites:
                if s["function"] in site_map:
                    site_map[s["function"]]["opt_mb"] = s["peak_mb"]
                    site_map[s["function"]]["opt_calls"] = s["calls"]
                else:
                    site_map[s["function"]] = {"base_mb": 0, "base_calls": 0,
                                               "opt_mb": s["peak_mb"], "opt_calls": s["calls"]}
            sorted_sites = sorted(site_map.items(),
                                  key=lambda x: max(x[1]["base_mb"], x[1]["opt_mb"]))
            syms = [k[:60] + "..." if len(k) > 60 else k for k, _ in sorted_sites]
            full_syms = [k for k, _ in sorted_sites]

            # Peak MB side-by-side (optimized first so baseline renders on top)
            traces = []
            if comparison and opt_sites:
                traces.append(go.Bar(
                    name=options.other_name, orientation="h",
                    x=[site_map[k]["opt_mb"] for k, _ in sorted_sites],
                    y=syms, marker_color=C_OTHER,
                    hovertemplate="%{customdata}<br>Peak: %{x:.2f} MB<extra></extra>",
                    customdata=full_syms,
                ))
            traces.append(go.Bar(
                name=options.baseline_name, orientation="h",
                x=[site_map[k]["base_mb"] for k, _ in sorted_sites],
                y=syms, marker_color=C_BASE,
                hovertemplate="%{customdata}<br>Peak: %{x:.2f} MB<extra></extra>",
                customdata=full_syms,
            ))
            fig = go.Figure(data=traces)
            fig.update_layout(
                title="Memory Hotspots (Peak MB): {}".format(bench_label),
                xaxis_title="Peak Consumption (MB)", barmode="group",
                yaxis=dict(automargin=True, tickfont=dict(size=11)),
                height=max(450, len(sorted_sites) * 30 + 120),
                margin=dict(l=0), yaxis_automargin=True,
            )
            hotspot_figs.append(fig)

            # Absolute allocation calls side-by-side
            if comparison and opt_sites:
                traces = [
                    go.Bar(
                        name=options.other_name, orientation="h",
                        x=[site_map[k]["opt_calls"] for k, _ in sorted_sites],
                        y=syms, marker_color=C_OTHER,
                        hovertemplate="%{customdata}<br>Calls: %{x}<extra></extra>",
                        customdata=full_syms,
                    ),
                    go.Bar(
                        name=options.baseline_name, orientation="h",
                        x=[site_map[k]["base_calls"] for k, _ in sorted_sites],
                        y=syms, marker_color=C_BASE,
                        hovertemplate="%{customdata}<br>Calls: %{x}<extra></extra>",
                        customdata=full_syms,
                    ),
                ]
                fig = go.Figure(data=traces)
                fig.update_layout(
                    title="Memory Hotspots (Allocation Calls): {}".format(bench_label),
                    xaxis_title="Calls", barmode="group",
                    yaxis=dict(automargin=True, tickfont=dict(size=11)),
                    height=max(450, len(sorted_sites) * 30 + 120),
                    margin=dict(l=0), yaxis_automargin=True,
                )
                hotspot_figs.append(fig)

    # ---- Write all figures: scatter first, then bars, then hotspots ----

    all_figs = scatter_figs + bar_figs + hotspot_figs
    title = (
        options.baseline_name
        if not comparison
        else "{0} vs {1}".format(options.baseline_name, options.other_name)
    )

    def _write_html(figs_list, path, page_title):
        with open(path, "w") as f:
            f.write("<html><head><title>{0}</title></head><body>\n".format(page_title))
            for fig in figs_list:
                f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write("</body></html>\n")
        print("  HTML: {0}".format(os.path.abspath(path)))

    # Full report
    report_path = os.path.join(options.output_dir, "profile_report.html")
    _write_html(all_figs, report_path, "Profile Report: " + title)

    # Summary report (scatter plots only) — useful when both perf and
    # heaptrack data exist in the same sim_run directory
    if scatter_figs:
        summary_path = os.path.join(options.output_dir, "profile_summary.html")
        _write_html(scatter_figs, summary_path, "Profile Summary: " + title)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def dump_summary(data, options):
    """Write summary text file and CSV to the output directory."""
    os.makedirs(options.output_dir, exist_ok=True)
    comparison = options.other_dir is not None

    # --- Summary text file ---
    summary_path = os.path.join(options.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Profile Comparison Summary\n")
        if comparison:
            f.write("Baseline: {} ({})\n".format(options.baseline_name, options.baseline_dir))
            f.write("Other:    {} ({})\n".format(options.other_name, options.other_dir))
        else:
            f.write("Run: {} ({})\n".format(options.baseline_name, options.baseline_dir))
        f.write("=" * 80 + "\n\n")

        # Per-app table
        if comparison:
            f.write("{:<60} {:>12} {:>12} {:>8} {:>8}\n".format(
                "Benchmark", "Base Rate", "Other Rate", "Speedup", "Cycles"))
            f.write("-" * 100 + "\n")
        else:
            f.write("{:<60} {:>12} {:>12}\n".format("Benchmark", "Rate (ips)", "Cycles"))
            f.write("-" * 84 + "\n")

        for entry in data:
            app = entry["base_sim"]["app"]
            rate = app.get("gpgpu_simulation_rate_ips", "N/A")
            cycles = app.get("gpu_tot_sim_cycle", "N/A")
            if comparison:
                oapp = entry["opt_sim"]["app"]
                orate = oapp.get("gpgpu_simulation_rate_ips", "N/A")
                try:
                    speedup = "{:.3f}x".format(float(orate) / float(rate))
                except (ValueError, TypeError, ZeroDivisionError):
                    speedup = "N/A"
                cyc_match = "OK" if cycles == oapp.get("gpu_tot_sim_cycle", "") else "DIFF"
                f.write("{:<60} {:>12} {:>12} {:>8} {:>8}\n".format(
                    entry["key"][:60], str(rate), str(orate), speedup, cyc_match))
            else:
                f.write("{:<60} {:>12} {:>12}\n".format(
                    entry["key"][:60], str(rate), str(cycles)))

        # Heaptrack summary per app
        f.write("\n\nHEAP MEMORY\n")
        f.write("-" * 80 + "\n")
        if comparison:
            f.write("{:<50} {:>12} {:>12} {:>12} {:>12}\n".format(
                "Benchmark", "Base Heap", "Other Heap", "Base Allocs", "Other Allocs"))
            f.write("-" * 98 + "\n")
        for entry in data:
            bht = entry.get("base_heaptrack", {})
            if not bht:
                continue
            peak = bht.get("peak_heap", "N/A")
            allocs = bht.get("total_allocations", "N/A")
            if comparison:
                oht = entry.get("opt_heaptrack", {})
                opeak = oht.get("peak_heap", "N/A")
                oallocs = oht.get("total_allocations", "N/A")
                f.write("{:<50} {:>12} {:>12} {:>12} {:>12}\n".format(
                    entry["key"][:50], str(peak), str(opeak), str(allocs), str(oallocs)))
            else:
                f.write("{:<50} {:>12} {:>12}\n".format(
                    entry["key"][:50], str(peak), str(allocs)))

        # Top CPU functions (first 5 benchmarks)
        f.write("\n\nTOP CPU FUNCTIONS\n")
        f.write("-" * 80 + "\n")
        for entry in data[:5]:
            funcs = entry.get("base_perf", {}).get("functions", [])
            if funcs:
                f.write("\n{} ({}):\n".format(entry["key"], options.baseline_name))
                for func in funcs[:10]:
                    f.write("  {:>6.1f}% {:>9} samples  {}\n".format(
                        func["overhead_pct"], func["samples"], func["symbol"]))
            if comparison:
                ofuncs = entry.get("opt_perf", {}).get("functions", [])
                if ofuncs:
                    f.write("\n{} ({}):\n".format(entry["key"], options.other_name))
                    for func in ofuncs[:10]:
                        f.write("  {:>6.1f}% {:>9} samples  {}\n".format(
                            func["overhead_pct"], func["samples"], func["symbol"]))

    print("Summary: {}".format(os.path.abspath(summary_path)), file=sys.stderr)

    # --- CSV for further analysis ---
    csv_path = os.path.join(options.output_dir, "summary.csv")
    with open(csv_path, "w") as f:
        if comparison:
            f.write("benchmark,base_rate_ips,other_rate_ips,speedup,"
                    "base_cycles,other_cycles,cycles_match,"
                    "base_peak_heap,other_peak_heap,"
                    "base_allocs,other_allocs\n")
        else:
            f.write("benchmark,rate_ips,cycles,peak_heap,allocs\n")

        for entry in data:
            app = entry["base_sim"]["app"]
            rate = app.get("gpgpu_simulation_rate_ips", "")
            cycles = app.get("gpu_tot_sim_cycle", "")
            bht = entry.get("base_heaptrack", {})
            peak = bht.get("peak_heap", "")
            allocs = bht.get("total_allocations", "")

            if comparison:
                oapp = entry["opt_sim"]["app"]
                orate = oapp.get("gpgpu_simulation_rate_ips", "")
                ocycles = oapp.get("gpu_tot_sim_cycle", "")
                try:
                    speedup = "{:.4f}".format(float(orate) / float(rate))
                except (ValueError, TypeError, ZeroDivisionError):
                    speedup = ""
                cyc_match = "1" if cycles == ocycles else "0"
                oht = entry.get("opt_heaptrack", {})
                opeak = oht.get("peak_heap", "")
                oallocs = oht.get("total_allocations", "")
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    entry["key"], rate, orate, speedup,
                    cycles, ocycles, cyc_match,
                    peak, opeak, allocs, oallocs))
            else:
                f.write("{},{},{},{},{}\n".format(
                    entry["key"], rate, cycles, peak, allocs))

    print("CSV:     {}".format(os.path.abspath(csv_path)), file=sys.stderr)


def main():
    options = parse_options()
    comparison = options.other_dir is not None

    # Resolve heaptrack_print once before spawning threads (avoids
    # AppImage extraction race conditions when using --appimage-extract-and-run)
    resolve_heaptrack_print(options.heaptrack_bin)

    baseline_benchmarks = discover_benchmarks(options.baseline_dir)
    if not baseline_benchmarks:
        sys.exit("No benchmark directories found in {0}".format(options.baseline_dir))

    if comparison:
        other_benchmarks = discover_benchmarks(options.other_dir)
        other_map = {key: path for key, path in other_benchmarks}
    else:
        other_map = {}

    def _parse_one(args):
        key, base_path, opt_path, heaptrack_bin = args
        entry = {
            "key": key,
            "base_perf": parse_perf_report(
                os.path.join(base_path, common.PROFILE_PERF_DATA_FILE)
            ),
            "base_heaptrack": parse_heaptrack(
                os.path.join(base_path, common.PROFILE_HEAPTRACK_FILE + ".zst"),
                heaptrack_bin,
            ),
            "base_sim": parse_sim_stats(base_path),
        }
        if opt_path:
            entry["opt_perf"] = parse_perf_report(
                os.path.join(opt_path, common.PROFILE_PERF_DATA_FILE)
            )
            entry["opt_heaptrack"] = parse_heaptrack(
                os.path.join(opt_path, common.PROFILE_HEAPTRACK_FILE + ".zst"),
                heaptrack_bin,
            )
            entry["opt_sim"] = parse_sim_stats(opt_path)
        elif comparison:
            entry["opt_perf"] = {"functions": []}
            entry["opt_heaptrack"] = {}
            entry["opt_sim"] = {"app": {}, "kernels": []}
        return entry

    work_items = [
        (key, base_path,
         other_map.get(key) if comparison else None,
         options.heaptrack_bin)
        for key, base_path in baseline_benchmarks
    ]

    with ThreadPoolExecutor(max_workers=options.jobs) as pool:
        futures = {pool.submit(_parse_one, w): w for w in work_items}
        data = []
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Parsing benchmarks"):
            data.append(future.result())
    # Restore original benchmark order (as_completed returns in finish order)
    key_order = [k for k, _ in baseline_benchmarks]
    data.sort(key=lambda e: key_order.index(e["key"]))

    generate_text_report(data, options)
    generate_plots(data, options)
    dump_summary(data, options)


if __name__ == "__main__":
    main()
