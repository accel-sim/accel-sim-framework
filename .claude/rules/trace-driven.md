---
description: Accel-Sim trace-driven wrapper source code
paths:
  - "gpu-simulator/**"
  - "!gpu-simulator/gpgpu-sim/**"
---

# Accel-Sim Wrapper Layer

The Accel-Sim wrapper layer sits on top of GPGPU-Sim and provides trace-driven simulation capabilities.

## Source Code Organization

### Entry Points

**`accel-sim.cc` / `accel-sim.h`**
- `accel_sim_framework` - Main orchestrator class
- Manages simulation loop, kernel launching, and trace parsing
- Key methods:
  - `simulation_loop()` - Main loop processing trace commands
  - `simulate()` - Advances simulation cycles
  - `create_kernel_info()` - Creates kernel from trace info
  - `gpgpu_trace_sim_init_perf_model()` - Initializes the simulator

**`main.cc`**
- CLI entry point, instantiates `accel_sim_framework`

### Trace-Driven Classes (`trace-driven/`)

**`trace_driven.h` / `trace_driven.cc`**
- Extends GPGPU-Sim base classes for trace replay

**Key Classes:**

| Class | Extends | Purpose |
|-------|---------|---------|
| `trace_gpgpu_sim` | `gpgpu_sim` | Top-level trace simulator |
| `trace_simt_core_cluster` | `simt_core_cluster` | Cluster with trace cores |
| `trace_shader_core_ctx` | `shader_core_ctx` | Core that replays traces instead of executing PTX |
| `trace_shd_warp_t` | `shd_warp_t` | Warp with trace replay state |
| `trace_warp_inst_t` | `warp_inst_t` | Instruction parsed from trace |
| `trace_kernel_info_t` | `kernel_info_t` | Kernel metadata from trace |
| `trace_function_info` | `function_info` | Stub function info for trace mode |
| `trace_config` | - | Trace-specific config (latencies, trace file path) |

**`trace_shader_core_ctx` Key Methods:**
- `get_next_inst()` - Returns next instruction from trace (not I-cache)
- `init_warps()` / `init_traces()` - Loads warp traces from file
- `func_exec_inst()` - Skips functional execution (trace provides results)
- `issue_warp()` - Issues trace instruction to execution unit

### Trace Parser (`trace-parser/`)

**`trace_parser.h` / `trace_parser.cc`**
- Parses NVBit-generated trace files

**Key Structures:**

`inst_trace_t` - Single instruction from trace:
- `m_pc` - Program counter
- `mask` - Active thread mask
- `opcode` - SASS opcode string
- `reg_src[]` / `reg_dest[]` - Register operands
- `memadd_info` - Memory addresses (for loads/stores)
- `tma_memadd_info` - TMA (Tensor Memory Access) info for Hopper+

`kernel_trace_t` - Kernel metadata:
- Grid/block dimensions, shared memory, registers
- `cuda_stream_id` - For concurrent kernel support
- `pipeReader` - Reads compressed trace file

`trace_command` - Command from trace commandlist:
- `kernel_launch`, `cpu_gpu_mem_copy`, `gpu_cpu_mem_copy`

`trace_parser` class:
- `parse_commandlist_file()` - Parses kernel launch sequence
- `parse_kernel_info()` - Reads kernel header from trace file
- `get_next_threadblock_traces()` - Streams threadblock traces

**Address Compression Formats:**
- `list_all` - All 32 addresses listed
- `base_stride` - Base + stride pattern
- `base_delta` - Base + per-thread deltas
- `tma_*` - TMA-specific formats (Hopper)

### ISA Definitions (`ISA_Def/`)

**`trace_opcode.h`**
- `TraceInstrOpcode` enum - All SASS opcodes across architectures
- `OpcodeChar` struct - Maps opcode string to execution unit type
- `get_OpcodeMap()` - Returns arch-specific opcode mapping

**Architecture-Specific Headers:**
- `kepler_opcode.h`, `pascal_opcode.h`, `volta_opcode.h`
- `turing_opcode.h`, `ampere_opcode.h`, `hopper_opcode.h`

Each defines `initOpMap()` populating opcode→unit mappings:
```cpp
OpcodeChar{"FFMA", OP_FFMA, FP, 1, ...}
// opcode_name, enum, op_type, latency, ...
```

**`accelwattch_component_mapping.h`**
- Maps SASS opcodes to power model components

### Python Wrapper (`python_wrapper/`)

**`python_wrapper.cc` / `python_wrapper.h`**
- pybind11 bindings for `accel_sim_framework`
- Enables Python-driven simulation

### Configuration (`configs/`)

**`configs/tested-cfgs/<arch>/trace.config`**
- Trace-specific settings (latencies, trace file)
- Complements `gpgpusim.config` from GPGPU-Sim

Key options:
```
-trace_opcode_latency_initiation_int 4,1
-trace_opcode_latency_initiation_sp 4,1
-trace_opcode_latency_initiation_dp 64,64
-trace_opcode_latency_initiation_sfu 21,8
-trace_opcode_latency_initiation_tensor 32,32
```

## Simulation Flow

1. **Parse commandlist** - Read `kernelslist.g` for kernel launch order
2. **For each kernel launch:**
   - Parse kernel header (grid dims, shmem, regs)
   - Create `trace_kernel_info_t`
   - Distribute CTAs to `trace_shader_core_ctx` instances
3. **Per-cycle execution:**
   - `trace_shader_core_ctx::get_next_inst()` fetches from trace
   - Instruction flows through GPGPU-Sim pipeline
   - Memory addresses come from trace, not functional execution
4. **Kernel completion** - Move to next command

## Key Differences from PTX Mode

| Aspect | PTX Mode | Trace Mode |
|--------|----------|------------|
| Instruction source | I-cache + PTX parser | Trace file |
| Memory addresses | Computed functionally | From trace |
| Branch decisions | SIMT stack | Pre-recorded in trace |
| Execution | Full functional sim | Timing only |

## Common Modification Patterns

### Adding a New Architecture

1. Create `ISA_Def/<arch>_opcode.h` with `initOpMap()`
2. Add case in `trace_opcode.h::get_OpcodeMap()`
3. Create `configs/tested-cfgs/SM<ver>_<name>/trace.config`

### Adding a New Opcode

1. Add to `TraceInstrOpcode` enum in `trace_opcode.h`
2. Add mapping in relevant `<arch>_opcode.h::initOpMap()`
3. If new execution unit, update `trace_config::set_latency()`

### Modifying Trace Parsing

- Address formats: `inst_memadd_info_t::base_*_decompress()`
- Instruction parsing: `inst_trace_t::parse_from_string()`
- Kernel header: `kernel_trace_t` constructor