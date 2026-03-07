---
description: Memory system (caches, DRAM, interconnect)
paths:
  - "gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.*"
  - "gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.*"
  - "gpu-simulator/gpgpu-sim/src/gpgpu-sim/dram.*"
  - "gpu-simulator/gpgpu-sim/src/gpgpu-sim/mem_fetch.*"
  - "gpu-simulator/gpgpu-sim/src/gpgpu-sim/icnt_wrapper.*"
  - "gpu-simulator/gpgpu-sim/src/gpgpu-sim/local_interconnect.*"
  - "gpu-simulator/gpgpu-sim/src/gpgpu-sim/mem_latency_stat.*"
---

# Memory System

## Memory Request Path

```
ldst_unit creates mem_fetch
    ↓
L1 Cache (baseline_cache::access())
    ↓ miss
Interconnect (icnt_wrapper::push())
    ↓
L2 Cache (l2_cache::access())
    ↓ miss
DRAM (dram_t::push())
    ↓
Response path back to shader
```

## Key Classes

### mem_fetch (mem_fetch.h:54)
Memory request packet flowing through hierarchy.
- Address and access type (read/write)
- Request metadata (warp ID, shader ID, timestamps)
- Status tracking through pipeline

### mem_access_t (abstract_hardware_model.h:879)
Memory access descriptor embedded in mem_fetch.
- Address, size, type (load/store/atomic)

## Caches

### baseline_cache (gpu-cache.h)
L1 instruction/data caches.
- `access()` - Main cache access method
- Tag arrays, replacement policies (LRU)
- MSHR tracking

### l2_cache (l2cache.h)
L2 cache with multiple sub-partitions.

### Modifying Cache Behavior
1. Policy changes in `gpu-cache.cc`
   - Replacement: `tag_array::access()`
   - Write policy: `baseline_cache::access()`
2. New cache type: inherit from `cache_t` or `baseline_cache`
3. Add config parameters

### MSHR Management
- L1/L2 have limited MSHRs
- Track outstanding requests
- Full MSHRs block new requests

## DRAM (dram.h)

`dram_t` - DRAM controller.
- Bank conflict modeling
- Row buffer hits/misses
- Timing: tRCD, tCAS, tRP, tRC

## Interconnect

### icnt_wrapper (icnt_wrapper.h)
Interface layer to interconnect.
- `push()` - Send packet
- `pop()` - Receive packet

### local_interconnect (local_interconnect.h)
Simple crossbar model.
- `Push()`, `Pop()`, `Advance()`

### Modifying Interconnect
1. Local crossbar: modify `local_interconnect.cc`
2. Global: modify `icnt_wrapper.cc`
3. New topology: changes in both + config parsing

## Config Parameters

```
-gpgpu_cache:il1 <config_string>
-gpgpu_cache:dl1 <config_string>
-gpgpu_dram_timing_opt <timing_params>
```
