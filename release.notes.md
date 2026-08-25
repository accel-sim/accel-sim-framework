
# Release v2.0.0 — Accel-Sim 2.0 (full Hopper support)

_TBD 2026._

Accel-Sim 2.0 brings **full NVIDIA Hopper (H100 / H200) support** to the trace-driven
simulator, modeling the asynchronous, warp-specialized, persistent execution style of
modern AI kernels — the Tensor Memory Accelerator (TMA), asynchronous Warp Group MMA
(WGMMA), `mbarrier`-based producer/consumer synchronization, and threadblock clusters
(Cooperative Groups). It adds a partitioned (chiplet / uGPU) memory subsystem, a rebuilt
tracer with a compressed trace format and PyTorch per-layer hooks, and **GPUVision**, a
CUPTI-based tool for cycle-level hardware correlation.

> **Roadmap:** experimental **Blackwell (B200 / RTX 5090)** support and **multi-GPU / NVLink**
> support are incoming shortly.

Work that uses any part of Accel-Sim should cite both the Accel-Sim 1.0 and 2.0 papers (plus
GPGPU-Sim) — see the "How to Cite" section in the [README](README.md) for the full guide.

## Table 1 — Hopper H100: Accel-Sim 1.x baseline vs. features added in Accel-Sim 2.0

| Component | Inherited from Accel-Sim 1.x | Added in Accel-Sim 2.0 |
|---|---|---|
| **# SMs** | 132 | — |
| **Operand Collector** | baseline | new operand collector, spinloop support, chiplet-aware CTA scheduling, Cooperative Groups (cluster) support, cluster control |
| **L1 Cache / Shared Mem** | 256 KB, 4 banks | TMA, CGA multicast |
| **# Exec Units** | 4 FP, 4 DP, 4 INT, 4 SFU, 4 HMMA | WGMMA, UTC\*MMA, warpgroup commit/wait, 2-CTA, exec-unit refactor |
| **L2 Cache** | 50 MB, 80 banks | modulo IPOLY hash, LRC, chiplet cache policies |
| **MMA Latency** | Fixed | Variable (depends on N) |
| **Memory** | HBM bandwidth / latency | HBM3, HBM3e bandwidth / latency |
| **Interconnect** | Monolithic crossbar | chiplet interconnect, NVLink |
| **Sync Primitives** | Barrier | `mbarrier`, `bar.arv`, `bar.sync`, `ldgstsbar`, `fence`, UTCBAR, WARPGROUP |
| **Tracer** | SASS register deps, control flow | register **value** tracing, spinloop handling, multi-GPU trace, trace compression, trace paging, tensor-descriptor support, PyTorch per-layer support |
| **Performance Counters** | 54 | **11,072** |
| **Correlation** | kernel-level Nsight Compute | cycle-level **GPUVision** |

## Major changes since v1.3.0

### 1. Full Hopper support — asynchronous Tensor Cores & synchronization
- **WGMMA / async Warp Group MMA** modeling: commit groups, `warpgroup.depbar`, wait-group
  semantics, and variable MMA latency that depends on the `N` tile dimension.
- **`mbarrier` producer/consumer synchronization**: async proxy fence, remote `mbarrier`
  arrive, kernel-level hashmap lookup for fast barrier state, writeback-completion fixes.
- **Barrier family**: `bar.arv` / `bar.sync`, `ARRIVES.LDGSTSBAR.64.ARVCNT`, UTCBAR, and two
  rounds of barrier-correctness fixes.
- These features are **automatic** under an H100 / H200 config — just trace and run.

### 2. TMA & threadblock clusters
- **TMA (Tensor Memory Accelerator)** bulk data movement: opcode-based detection, bulk
  group, store group, LDGSTS `mbarrier`-based completion, out-of-bound byte tracking.
- **Threadblock clusters (CGA)** with shared-memory multicast: cluster-info parsing,
  multicast masks, and TMA multicast to all SMs in a cluster.

### 3. Spinloop / asynchronous-wait modeling
- The simulator now **dynamically re-evaluates** `NANOSLEEP` / `TRYWAIT` spinloops instead
  of replaying a fixed, unrepresentative number of polling iterations captured at trace
  time. The tracer filters redundant polling iterations to a single canonical pass, and
  the simulator natively reproduces the wait behavior — enabling faithful synchronization
  overhead studies.

### 4. Chiplet / uGPU partitioned memory subsystem
- **Chiplet interconnect** with latency-modeled inter-chiplet queues, plus `CHIPLET_ACC`
  access types for inter-chiplet traffic accounting.
- **L2 Request Coalescer (LRC)** modeling the H100's request-merge behavior (configurable
  merge ratio), with dedicated hardware counters.
- **IPOLY + MODULO L2 hashing** (`-gpgpu_memory_partition_indexing 6`) for non-power-of-two
  bank/subpartition counts (e.g. H200).
- Chiplet cache policies (local write-through / remote write-back).

### 5. Rebuilt tracer
- Upgraded to **public NVBit v1.8**.
- **Register value tracing** for tensor descriptors and `mbarrier` operands.
- **Compressed trace format** — per-warp zstd `.tracez` — plus **page-based trace loading**
  that bounds simulator runtime memory to ~4 GB regardless of kernel size.
- **Multithreaded post-processing** (`-j`).
- **PyTorch per-layer tracer hook** — selectively trace individual model layers (vLLM /
  Hugging Face), toggling NVBit instrumentation per forward pass.

### 6. GPUVision — cycle-level CUPTI profiling
- A **CUPTI PM-sampling** tool that collects hardware performance counters as a cycle-level
  time series (not a single per-kernel aggregate), replacing kernel-level AerialVision.
  Supports CUDA Graphs, per-device output, and Nsight Compute replay + cache-control modes.
  This enables **cycle-level** correlation of the simulator against real hardware.

### 7. Redesigned statistics & simulator performance
- A rebuilt statistics subsystem exposing **11,072 counters** (up from 54), with chiplet
  stat merging and fast CSV parsing (stats collection that previously took hours completes
  in about a minute).
- Operand-collector and SIMT-pipeline refactor plus a `simple_dram` fast path, improving
  simulation throughput to ~27.5K KIPS (about 2.2× faster than Accel-Sim 1.x).

### 8. Configurations, apps & job launching
- **Tested configs** for H100 (`SM90_H100`) and H200 (`SM90_H200`).
- `--per-kernel` parallel kernel execution, per-app `kernel-name-filter`, and replay-region
  support in `run_simulations.py`.
- Modern ML workload definitions (LLM inference/training, CUTLASS, FlashAttention-3, DLRM).

## Upgrading from Accel-Sim 1.x
- **New trace format:** post-processing now emits compressed `.tracez` by default; pass
  `--text` to `post-traces-processing` for the legacy plain `.traceg`. Use the `traceDsm`
  tool to decode `.tracez` back to `.traceg` (see the README).
- **New configs:** use `-C H100-SASS` or `-C H200-SASS`; older configs (QV100, A100, …) are
  unchanged.
- **Performance model:** built on the GPGPU-Sim 4.x performance model updated for Hopper.

# Release v1.3.0

Oct 19, 2021.

Major changes since v1.2.0:

1. Release and integration of AccelWattch v1.0 with Accel-Sim.

# Release v1.2.0

May 26, 2021.

Major changes since v1.1.0:

1. First release of Accel-Sim's tuner component
2. Nvbit 1.5.2 support
3. Partial support of cutting-edge MLPerf traces
4. Turing architecture traces
5. Ampere architecture traces (without copy-async feature: still in progress)
6. GPGPU-Sim 4.1 release
7. Detailed manual

# Release v1.1.0

Oct 23, 2020.

Major changes since v1.0.0:

1. Advanced disk space format to reduce the traces size (up to 5-10X reduction).
2. Optimized runtime memory footprint (up to 5-10X reduction).
3. Support for Nvbit 1.5
4. Fixed the generic loads address mapping
5. Fixed atomic instruction parsing bug
6. Fixed the max_insn_limit bug
7. Adding support to Turing instruction traces

---
