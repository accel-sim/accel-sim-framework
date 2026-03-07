---
description: Shader core pipeline and execution units
paths:
  - "gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.*"
  - "gpu-simulator/gpgpu-sim/src/gpgpu-sim/scoreboard.*"
---

# Shader Core (shader.h/cc)

The most complex files in GPGPU-Sim.

## Key Classes

### shader_core_ctx
Single SM (Streaming Multiprocessor). Manages:
- Instruction fetch, decode, issue, execute, writeback
- Warp schedulers
- Execution units (SP, SFU, DP, INT, TENSOR, MEM)
- Operand collectors and register files
- L1 caches and shared memory
- SIMT stacks

**Main method:** `shader_core_ctx::cycle()`

### simt_core_cluster
Container for cluster of shader cores. Manages local interconnect.

### shd_warp_t
Warp state: PC, active threads, functional state.

### thread_ctx_t
Per-thread context and statistics.

## Pipeline Stages

Called in reverse order in `cycle()`:

1. **writeback()** - Write results to register file
2. **execute()** - Send to execution units
3. **issue()** - Check operand collectors, issue ready instructions
4. **decode()** - Decode fetched instructions
5. **fetch()** - I-cache access

## Schedulers

Base: `scheduler_unit`

| Scheduler | Description |
|-----------|-------------|
| `lrr_scheduler` | Loose round-robin |
| `gto_scheduler` | Greedy then oldest |
| `oldest_scheduler` | Oldest first |
| `two_level_active_scheduler` | Two-level |

### Modifying Schedulers
1. Create class inheriting `scheduler_unit` in shader.h
2. Implement `cycle()` to select warps
3. Register in `shader_core_ctx::create_schedulers()`
4. Add config option in `shader_core_config`

## Execution Units

Base: `simd_function_unit`, `pipelined_simd_unit`

| Unit | Purpose |
|------|---------|
| `sp_unit` | Single-precision FP |
| `sfu` | Special functions (sin, cos, sqrt) |
| `dp_unit` | Double-precision FP |
| `int_unit` | Integer ALU |
| `tensor_core` | Tensor/matrix ops |
| `ldst_unit` | Load/store memory |
| `specialized_unit` | Architecture-specific |

## Operand Collector

`opndcoll_rfu_t` - Collects operands from register file.
- Manages register bank conflicts
- `allocate_reads()`, `allocate_writes()`

## Config Parameters

```
-gpgpu_n_cores <N>
-gpgpu_n_clusters <N>
-gpgpu_shader_core_pipeline "2048:32:32:32"
-gpgpu_num_sp_units <N>
-gpgpu_num_sfu_units <N>
```
