---
description: GPGPU-Sim overview and general guidance
paths:
  - "gpu-simulator/gpgpu-sim/**"
---

# GPGPU-Sim

Core GPU performance simulator. 159 source files under `src/`.

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Top-level infrastructure (`abstract_hardware_model`, `gpgpusim_entrypoint`) |
| `src/gpgpu-sim/` | Performance/timing model (shader cores, caches, DRAM) |
| `src/cuda-sim/` | Functional simulation (PTX execution) |
| `src/accelwattch/` | Power modeling |
| `src/intersim2/` | Detailed interconnect (BookSim-based) |

## Formatter

```bash
./gpu-simulator/gpgpu-sim/format-code.sh
```

## Main Simulation Loop

```
gpgpu_sim::cycle()
├── shader cores: shader_core_ctx::cycle()
├── interconnect: icnt_interface::transfer()
└── memory partitions: memory_partition_unit::cycle()
    ├── L2 caches: l2_cache::cycle()
    └── DRAM: dram_t::cycle()
```

## Config System

- Format: `-option_name value` in `gpgpusim.config`
- Parsed by `option_parser.h/cc`
- Key classes: `gpgpu_sim_config`, `shader_core_config`, `memory_config`

## Adding Statistics

Use `/add-counter` skill for step-by-step guidance.
