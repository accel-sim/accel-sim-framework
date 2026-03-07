---
description: Core abstractions and top-level simulator
paths:
  - "gpu-simulator/gpgpu-sim/src/abstract_hardware_model.*"
  - "gpu-simulator/gpgpu-sim/src/gpgpusim_entrypoint.*"
  - "gpu-simulator/gpgpu-sim/src/stream_manager.*"
  - "gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.*"
---

# Core Classes

## abstract_hardware_model.h

Base classes used throughout the simulator.

### inst_t
Base instruction class.
- Opcode, source/destination operands, PC

### warp_inst_t
**Central instruction structure** - represents a warp instruction.
- Extends `inst_t`
- Active thread mask
- Memory addresses (for load/store)
- Latency and issue cycle tracking
- Execution unit type (`exec_unit_type_t`: SP, SFU, DP, INT, TENSOR, MEM)

### kernel_info_t
Kernel metadata.
- Entry point, PTX code, parameter memory
- Grid/block dimensions, shared memory size
- CTA distribution management

### simt_stack
Branch divergence tracking.
- Stack-based reconvergence
- Each warp has a SIMT stack
- `launch()`, `update()` for branches

## gpu-sim.h/cc

### gpgpu_sim
Main simulator orchestrator.
- Contains: `simt_core_cluster[]`, `memory_partition_unit[]`, interconnect
- **Key method:** `cycle()` - advances all components one cycle
- Manages concurrent kernels

### gpgpu_sim_config
Top-level configuration.
- Inherits from `power_config`, `gpgpu_functional_sim_config`

### memory_config
DRAM timing, addressing, interconnect topology.

## gpgpusim_entrypoint.cc

Simulator initialization and main entry point.
- Config file parsing
- Create `gpgpu_sim` instance

## stream_manager.cc

CUDA stream and kernel launch management.
- `register_kernel()` - register kernel for execution
- Distribute CTAs to shader cores
