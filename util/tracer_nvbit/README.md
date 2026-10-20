# 📘 Using the Tracer Tool for Accel-Sim

This document explains how to use the `tracer_tool` for generating instruction traces for GPU applications. The tool supports full benchmark suites, individual applications, specific kernel tracing, source line mapping, and more.

---

## 🛠️ Setup and Installation

Before using the tracer, make sure to install and build the required tools:

```bash
# Install NVBit
./install_nvbit.sh

# Compile tracer tools
./make
```

---

## ⚙️ Option 1: Trace a Full Benchmark Suite

Use this if you're tracing an entire benchmark suite (e.g., Rodinia):

```bash
./run_hw_trace.py -B rodinia-3.1 -D 0
```

- `-B`: Benchmark suite name  (app list can be found or defined in [this file](../job_launching/apps/define-all-apps.yml))
- `-D`: Hardware device ID (e.g., GPU 0), a comma separated list (`-D 0,1,2,3`), or `all`

📁 Traces will be stored in:  
`../../hw_run/traces/device-0/` (with several GPUs, `device-<first listed GPU>`)

**This script handles trace generation, post-processing, and cleanup automatically.**  

On a node with several GPUs, the apps are spread over two worker pools:

```bash
./run_hw_trace.py -B rodinia-3.1 -D 0,1,2,3 --cpu_workers 4 -j 8
```

- One GPU tracing worker per GPU in `-D`, each with that GPU to itself, takes the next app of the suites in order and traces it with NVBit.
- A traced app goes on a queue; `--cpu_workers` post-processing workers take apps off it and post-process them with `-j` threads each, while the GPUs keep tracing.
- `--queue_size` (default: the number of GPUs) is how many traced apps may wait for a post-processing worker. When the queue is full, a GPU worker that finished a trace waits for room.
- Every output line is prefixed with the worker and app (`[gpu2 l1_lat/...]`, `[cpu0 ...]`). A failed app does not stop the others; the failures are listed at the end and the script exits non-zero.
- Each app's directory has `traced_on_gpu.txt` (index, name, UUID, PCI bus id of the GPU that traced it), `run_trace.sh` (tracing), `run_postprocess.sh` (post-processing), and `run.sh`, which runs both.  
See the [Trace File Structure](#-trace-file-structure) section for details on the output.

---

## ⚙️ Option 2: Trace an Individual Application

Use this approach if you want to trace a specific application binary (e.g., `vectoradd`):

```bash
export CUDA_VISIBLE_DEVICES=0
LD_PRELOAD=./tracer_tool/tracer_tool.so ./nvbit_release/test-apps/vectoradd/vectoradd
```

📁 Traces will appear in the `traces/` folder.  
See the [Trace File Structure](#-trace-file-structure) section for contents.  
🔄 **Note:** Unlike Option 1, **you must manually perform post-processing**:

```bash
./tracer_tool/traces-processing/post-traces-processing ./traces/kernelslist
```

This will generate `.traceg` and `kernelslist.g` files that are ready for Accel-Sim.

---

## 📦 Trace File Structure

Each trace folder contains:

- `kernel-*.trace`: Raw trace files (one per kernel)
- `kernelslist`: List of traced kernels and CUDA memcpy operations
- `stats.csv`: Summary statistics (instruction counts, kernel IDs, etc.)

After post-processing:
- `.traceg`: Grouped trace files by thread block
- `kernelslist.g`: Final trace list for use with Accel-Sim

---

## 🎯 Selective Tracing (Kernel-Based Filtering)

You can now trace specific kernels using the `DYNAMIC_KERNEL_RANGE` environment variable.

### Usage

Set `DYNAMIC_KERNEL_RANGE` to specify which kernel IDs (and optionally names) to trace. Supported formats:

- **Single ID:**  
  ```bash
  export DYNAMIC_KERNEL_RANGE="3"
  ```
  Traces only kernel 3.

- **Range:**  
  ```bash
  export DYNAMIC_KERNEL_RANGE="5-8"
  ```
  Traces kernels 5 through 8 (inclusive).

- **Open-ended Range:**  
  ```bash
  export DYNAMIC_KERNEL_RANGE="10-"
  ```
  Traces from kernel 10 onward.

- **Multiple Ranges (space-separated):**  
  ```bash
  export DYNAMIC_KERNEL_RANGE="2 5-8 10-"
  ```

- **With Name Filters (regex):**  
  ```bash
  export DYNAMIC_KERNEL_RANGE="5-8@kernel_a.*,kernel_b.*"
  ```
  Traces kernels 5–8 *only* if their names match `kernel_a.*` or `kernel_b.*`.

### To Disable Tracing But Still List Kernels

To list kernel metadata in `stats.csv` without generating traces, we set the DYNAMIC_KERNEL_RANGE to very large number  :
```bash
export DYNAMIC_KERNEL_RANGE="1000000"
```

This is useful for discovering kernel IDs and names without producing large trace files.


---

## ⏱️ Alternative: Trace Specific Code Regions with CUDA Profiling

Wrap regions to trace using CUDA APIs:
```cpp
cudaProfilerStart();
// region to trace
cudaProfilerStop();
```

Then disable default tracing:
```bash
export ACTIVE_FROM_START=0
```

---

## 🔍 Trace Source Line Mapping

Enable source line information in your traces:

1. Set environment variable:
```bash
export TRACE_LINEINFO=1
```

2. Rebuild benchmark applications:
```bash
source ./gpu-app-collection/src/setup_environment
make -j -C ./gpu-app-collection/src rodinia_2.0-ft
```

Traces will now include line number info from the original CUDA source (requires `-lineinfo` flag in NVCC).

---

## 📄 Trace Format Explanation

Each instruction has at least 10 required columns:

```
[line_num] PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [addresscompress?] [mem_addresses]
```

Details:
- Fields in `[]` are optional and appear only if applicable.
- `dest_num = 0` → no destination register field.
- `mem_width = 0` → no memory address info present.

🧾 **Example:**
```
31 0 0 3 0000 ffffffff 1 R1 IMAD.MOV.U32 2 R255 R255 0
```

This line represents:
- Threadblock: (31, 0, 0)  
- PC: `0000`, Mask: `ffffffff`  
- One destination register: `R1`  
- Opcode: `IMAD.MOV.U32`  
- Two source registers: `R255`, `R255`  
- Not a memory instruction (`mem_width = 0`)
