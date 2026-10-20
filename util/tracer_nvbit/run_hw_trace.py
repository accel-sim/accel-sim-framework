#!/usr/bin/env python3

from optparse import OptionParser
import os
import subprocess
import os
import queue
import threading
import time

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
import sys

sys.path.insert(0, os.path.join(this_directory, "..", "job_launching"))
import common
import re
import shutil
import glob
import datetime
import yaml
import common
import re
import datetime

# We will look for the benchmarks
parser = OptionParser()
parser.add_option(
    "-B",
    "--benchmark_list",
    dest="benchmark_list",
    help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for "
    + "the benchmark suite names.",
    default="rodinia_2.0-ft",
)
parser.add_option(
    "-D",
    "--device_num",
    dest="device_num",
    help="CUDA device number(s): a comma separated list (e.g. 0,1,2,3) or 'all' (every GPU "
    + "nvidia-smi lists). One GPU tracing worker runs per listed GPU. Device numbers follow "
    + "nvidia-smi (PCI bus id) order. Traces go under device-<first listed GPU>.",
    default="0",
)
parser.add_option(
    "-n",
    "--norun",
    dest="norun",
    action="store_true",
    help="Do not actually run the apps, just create the dir structure and launch files",
)
parser.add_option(
    "-l",
    "--limit_kernel_number",
    dest="kernel_number",
    default=-99,
    help="Sets a hard limit to the number of traced kernels",
)
parser.add_option(
    "-t",
    "--terminate_upon_limit",
    dest="terminate_upon_limit",
    action="store_true",
    help="Once the kernel limit is reached, terminate the tracing process",
)
parser.add_option(
    "--spinlock_handling",
    dest="spinlock_handling",
    choices=["none", "fast_forward", "mark_region"],
    default="mark_region",
    help="How to handle spinlock instructions: none, fast_forward, or mark_region (default: mark_region)",
)
parser.add_option(
    "--spinlock_fast_forward_iterations",
    dest="spinlock_fast_forward_iterations",
    type=int,
    default=1,
    help="Number of iterations to keep for spinlock fast forwarding. Only used if spinlock_handling is fast_forward",
)
parser.add_option(
    "--output_name",
    dest="output_name",
    default="hw_run",
    help="Name of the output directory to save the traces",
)
parser.add_option(
    "-j",
    "--nproc",
    dest="nproc",
    default=8,
    help="Number of processes to use for post-processing traces",
)
parser.add_option(
    "--text",
    dest="text_traces",
    action="store_true",
    default=False,
    help="Post-process into the old text based trace (.traceg) instead of the default .tracez",
)
parser.add_option(
    "--cpu_workers",
    dest="cpu_workers",
    type=int,
    default=1,
    help="Number of post-processing workers. Each one post-processes one traced app at a "
    + "time with -j/--nproc threads, while the GPU tracing workers keep tracing.",
)
parser.add_option(
    "--queue_size",
    dest="queue_size",
    type=int,
    default=0,
    help="Number of traced apps that may wait for a post-processing worker. A GPU tracing "
    + "worker that finishes a trace while the queue is full waits for room. "
    + "Default: the number of GPUs.",
)

(options, args) = parser.parse_args()


# Each GPU is a dict with index, name, uuid and pci_bus_id. nvidia-smi is only
# required for -D all; without it the listed device numbers are used as given.
def get_gpus(device_spec):
    smi = {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,pci.bus_id",
                "--format=csv,noheader",
            ],
            universal_newlines=True,
        )
        for line in out.strip().splitlines():
            index, name, uuid, pci_bus_id = [f.strip() for f in line.split(",")]
            smi[int(index)] = {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "pci_bus_id": pci_bus_id,
            }
    except (OSError, subprocess.CalledProcessError):
        if device_spec == "all":
            exit("-D all needs nvidia-smi to list the GPUs")

    if device_spec == "all":
        indices = sorted(smi)
    else:
        indices = [int(d) for d in device_spec.split(",")]
    if len(set(indices)) != len(indices):
        exit(f"GPU listed twice in -D {device_spec}")

    gpus = []
    for index in indices:
        if smi and index not in smi:
            exit(f"GPU {index} not found by nvidia-smi")
        gpus.append(
            smi.get(index, {"index": index, "name": "", "uuid": "", "pci_bus_id": ""})
        )
    names = set(g["name"] for g in gpus)
    if len(names) > 1:
        exit(f"-D {device_spec} lists different GPU models: {', '.join(sorted(names))}")
    return gpus


gpus = get_gpus(options.device_num)
# All traces go under the first listed GPU, so -T paths stay the same as with a single -D
device_label = str(gpus[0]["index"])
queue_size = options.queue_size if options.queue_size > 0 else len(gpus)

common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cuda_version = common.get_cuda_version(this_directory)
now_time = datetime.datetime.now()
day_string = now_time.strftime("%y.%m.%d-%A")
time_string = now_time.strftime("%H:%M:%S")
logfile = day_string + "--" + time_string + ".csv"

nvbit_tracer_path = os.path.join(this_directory, "tracer_tool")
nvbit_spinlock_path = os.path.join(this_directory, "others", "spinlock_tool")

# Map spinlock handling mode to integer
spinlock_mode_map = {"none": 0, "fast_forward": 1, "mark_region": 2}


# The GPU-dependent scripts: run_trace.sh (spinlock detection + NVBit tracing),
# run_spinlock_detection.sh, and run.sh (run_trace.sh then run_postprocess.sh).
# They are written when a GPU tracing worker picks the app, with that worker's GPU.
def write_trace_scripts(item, gpu):
    sh_contents = item["sh_contents"]
    sh_contents += (
        '\nexport CUDA_VERSION="'
        + cuda_version
        + '"; export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES="'
        + str(gpu["index"])
        + '" ; '
    )

    # first we generate the traces (.trace and kernelslist files) in run_trace.sh
    # then, run_postprocess.sh does post-processing for the traces and generates (.tracez by default, or .traceg with --text, and kernelslist.g files)
    # and deletes the intermediate files ((.trace and kernelslist files files)
    tracer_contents = sh_contents
    if options.spinlock_handling in ["fast_forward", "mark_region"]:
        tracer_contents += "\nbash run_spinlock_detection.sh"
    tracer_contents += (
        "\nrm -f traces/*"
        + "\nexport TRACES_FOLDER="
        + item["run_dir"]
        + f"; SPINLOCK_HANDLING_MODE={spinlock_mode_map[options.spinlock_handling]} SPINLOCK_ITER_TO_KEEP={options.spinlock_fast_forward_iterations} CUDA_INJECTION64_PATH="
        + os.path.join(nvbit_tracer_path, "tracer_tool.so")
        + " "
        + item["exec_path"]
        + " "
        + str(item["args"])
        + " ; "
    )

    # Spinlock tool run script
    # will run twice, once for phase 0 and once for phase 1
    spinlock_contents = (
        sh_contents
        + "\nrm -f spinlock_detection/*"
        + "\nexport TRACES_FOLDER="
        + item["run_dir"]
        + "; SPINLOCK_PHASE=0 CUDA_INJECTION64_PATH="
        + os.path.join(nvbit_spinlock_path, "spinlock_tool.so")
        + " "
        + item["exec_path"]
        + " "
        + str(item["args"])
        + " ; "
        + " SPINLOCK_PHASE=1 CUDA_INJECTION64_PATH="
        + os.path.join(nvbit_spinlock_path, "spinlock_tool.so")
        + " "
        + item["exec_path"]
        + " "
        + str(item["args"])
        + " ; "
    )

    run_contents = "set -e\nbash run_trace.sh\nbash run_postprocess.sh\n"

    for path, content in [
        ("run_trace.sh", tracer_contents),
        ("run_spinlock_detection.sh", spinlock_contents),
        ("run.sh", run_contents),
    ]:
        write_script(item["run_dir"], path, content)


def write_script(run_dir, path, content):
    open(os.path.join(run_dir, path), "w").write(content)
    if subprocess.call(["chmod", "u+x", os.path.join(run_dir, path)]) != 0:
        exit(f"Error chmod {path} runfile")


# Set up every app's directory before any GPU work, so path errors show up first
items = []
for bench in benchmarks:
    edir, ddir, exe, argslist = bench
    for argpair in argslist:
        args = argpair["args"]
        kernel_name_filter = (
            argpair.get("kernel-name-filter", "") if isinstance(argpair, dict) else ""
        )
        run_name = os.path.join(exe, common.get_argfoldername(argpair))
        this_run_dir = os.path.abspath(
            os.path.expandvars(
                os.path.join(
                    this_directory,
                    "..",
                    "..",
                    options.output_name,
                    "traces",
                    "device-" + device_label,
                    cuda_version,
                    run_name,
                )
            )
        )
        this_trace_folder = os.path.join(this_run_dir, "traces")
        if not os.path.exists(this_run_dir):
            os.makedirs(this_run_dir)
        if not os.path.exists(this_trace_folder):
            os.makedirs(this_trace_folder)

        # link the data directory
        try:
            benchmark_data_dir = common.dir_option_test(
                os.path.join(ddir, exe, "data"), "", this_directory
            )
            if os.path.lexists(os.path.join(this_run_dir, "data")):
                os.remove(os.path.join(this_run_dir, "data"))
            os.symlink(benchmark_data_dir, os.path.join(this_run_dir, "data"))
        except common.PathMissing:
            pass

        all_data_link = os.path.join(this_run_dir, "data_dirs")
        if os.path.lexists(all_data_link):
            os.remove(all_data_link)
        top_data_dir_path = common.dir_option_test(ddir, "", this_directory)
        os.symlink(top_data_dir_path, all_data_link)

        if args == None:
            args = ""
        exec_path = common.file_option_test(os.path.join(edir, exe), "", this_directory)
        sh_contents = "set -e\n"

        if options.terminate_upon_limit:
            sh_contents += "export TERMINATE_UPON_LIMIT=1; "

        if "mlperf" in exec_path:
            # For mlperf by default we turn this flag on
            sh_contents += "export TERMINATE_UPON_LIMIT=0; "
            exec_path = ". " + exec_path

            if options.kernel_number > 0:
                sh_contents += (
                    '\nexport DYNAMIC_KERNEL_RANGE="0-'
                    + str(options.kernel_number)
                    + '"\n'
                )
            else:
                sh_contents += '\nexport DYNAMIC_KERNEL_RANGE="0-' + str(50) + '"\n'
        else:
            if kernel_name_filter:
                sh_contents += (
                    f'\nexport DYNAMIC_KERNEL_RANGE="@{kernel_name_filter}"\n'
                )
            elif options.kernel_number > 0:
                sh_contents += (
                    '\nexport DYNAMIC_KERNEL_RANGE="0-'
                    + str(options.kernel_number)
                    + '"\n'
                )
            else:
                sh_contents += f'\nexport DYNAMIC_KERNEL_RANGE="{os.environ.get("DYNAMIC_KERNEL_RANGE", "")}"\n'

        # Post-processing needs no GPU, so its script is the same whichever GPU traced the app
        postprocess_contents = (
            "set -e\n"
            + os.path.join(
                nvbit_tracer_path, "traces-processing", "post-traces-processing"
            )
            + " "
            + this_trace_folder
            + " -j "
            + str(options.nproc)
            + (" --text" if options.text_traces else "")
            + " ; rm -f "
            + this_trace_folder
            + "/*.trace ; rm -f "
            + this_trace_folder
            + "/*.trace.xz ; rm -f "
            + this_trace_folder
            + "/kernelslist "
        )
        write_script(this_run_dir, "run_postprocess.sh", postprocess_contents)

        item = {
            "name": run_name,
            "run_dir": this_run_dir,
            "exec_path": exec_path,
            "args": args,
            "sh_contents": sh_contents,
        }
        if options.norun:
            write_trace_scripts(item, gpus[0])
        items.append(item)

if options.norun:
    exit(0)


# Traced apps waiting for a post-processing worker. _put runs under the queue's
# lock, so max_len is the exact largest number of waiting apps.
class FinishedTracingQueue(queue.Queue):
    def _init(self, maxsize):
        queue.Queue._init(self, maxsize)
        self.max_len = 0

    def _put(self, item):
        queue.Queue._put(self, item)
        if item is not None:
            self.max_len = max(self.max_len, len(self.queue))


pending_tracing = queue.Queue()
for item in items:
    pending_tracing.put(item)
finished_tracing = FinishedTracingQueue(maxsize=queue_size)

print_lock = threading.Lock()
stats_lock = threading.Lock()
stats = {"traced_ok": 0, "trace_failed": 0, "pp_ok": 0, "pp_failed": 0}
stall_seconds = {}
failures = []
stop = threading.Event()


def log(prefix, msg):
    with print_lock:
        print(f"[{prefix}] {msg}", flush=True)


# Run one generated script in the app's directory; every output line is prefixed
def run_prefixed(script, run_dir, prefix):
    proc = subprocess.Popen(
        ["bash", script],
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in proc.stdout:
        log(prefix, line.decode(errors="replace").rstrip("\n"))
    return proc.wait()


def gpu_tracing_worker(gpu):
    stall_seconds[gpu["index"]] = 0.0
    while not stop.is_set():
        try:
            item = pending_tracing.get_nowait()
        except queue.Empty:
            return
        prefix = f"gpu{gpu['index']} {item['name']}"
        write_trace_scripts(item, gpu)
        open(os.path.join(item["run_dir"], "traced_on_gpu.txt"), "w").write(
            "index,name,uuid,pci_bus_id\n"
            + ",".join([str(gpu["index"]), gpu["name"], gpu["uuid"], gpu["pci_bus_id"]])
            + "\n"
        )
        log(prefix, "tracing")
        start = time.time()
        if run_prefixed("run_trace.sh", item["run_dir"], prefix) != 0:
            log(prefix, f"tracing FAILED after {time.time() - start:.1f} s")
            with stats_lock:
                stats["trace_failed"] += 1
                failures.append(("tracing", f"gpu{gpu['index']}", item["run_dir"]))
            continue
        log(prefix, f"tracing done in {time.time() - start:.1f} s")
        with stats_lock:
            stats["traced_ok"] += 1
        if finished_tracing.full():
            log(prefix, "queue full, waiting for a post-processing worker")
        start = time.time()
        finished_tracing.put(item)
        stall_seconds[gpu["index"]] += time.time() - start


def cpu_postprocessing_worker(worker_id):
    while True:
        item = finished_tracing.get()
        if item is None:
            return
        prefix = f"cpu{worker_id} {item['name']}"
        log(prefix, "post-processing")
        start = time.time()
        if run_prefixed("run_postprocess.sh", item["run_dir"], prefix) != 0:
            log(prefix, f"post-processing FAILED after {time.time() - start:.1f} s")
            with stats_lock:
                stats["pp_failed"] += 1
                failures.append(("post-processing", f"cpu{worker_id}", item["run_dir"]))
            continue
        log(prefix, f"post-processing done in {time.time() - start:.1f} s")
        with stats_lock:
            stats["pp_ok"] += 1


usable_cpus = len(os.sched_getaffinity(0))
if options.cpu_workers * int(options.nproc) > usable_cpus:
    print(
        f"Warning: {options.cpu_workers} post-processing workers x -j {options.nproc} "
        + f"threads is more than the {usable_cpus} CPUs this process may use"
    )
print(
    f"Tracing {len(items)} apps on GPUs {','.join(str(g['index']) for g in gpus)}; "
    + f"{options.cpu_workers} post-processing workers x -j {options.nproc}; "
    + f"queue size {queue_size}"
)

run_start = time.time()
gpu_threads = [
    threading.Thread(target=gpu_tracing_worker, args=(gpu,), daemon=True)
    for gpu in gpus
]
cpu_threads = [
    threading.Thread(target=cpu_postprocessing_worker, args=(i,), daemon=True)
    for i in range(options.cpu_workers)
]
try:
    for t in cpu_threads + gpu_threads:
        t.start()
    for t in gpu_threads:
        while t.is_alive():
            t.join(1)
    # One end marker per post-processing worker, after every traced app
    for _ in cpu_threads:
        finished_tracing.put(None)
    for t in cpu_threads:
        while t.is_alive():
            t.join(1)
except KeyboardInterrupt:
    stop.set()
    sys.exit(
        "Interrupted: no new apps are started; running ones are killed with this script"
    )

print(f"\nDone in {time.time() - run_start:.1f} s")
print(
    f"apps {len(items)}: traced ok {stats['traced_ok']}, tracing failed {stats['trace_failed']}; "
    + f"post-processed ok {stats['pp_ok']}, post-processing failed {stats['pp_failed']}"
)
print(
    f"largest number of traced apps waiting for post-processing: {finished_tracing.max_len} "
    + f"(queue size {queue_size})"
)
print(
    "GPU time waiting for queue room (s): "
    + ", ".join(f"gpu{g}={s:.1f}" for g, s in sorted(stall_seconds.items()))
)
if failures:
    print("\nFailed apps:")
    for stage, worker, run_dir in failures:
        print(f"  {stage:16s} {worker:6s} {run_dir}")
    sys.exit(1)
