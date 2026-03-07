---
name: add-counter
description: Add a new statistics counter to GPGPU-Sim. Guides through declaring, initializing, incrementing, and registering metric counters.
---

When the user invokes this skill, help them add a new statistics counter to GPGPU-Sim.

Arguments: $ARGUMENTS

## Overview

GPGPU-Sim uses a statistics framework in `gpu-simulator/gpgpu-sim/src/statistics.h`. Reference implementation: `chiplet_queue_full`.

## Counter Types

Available in `Statistics::` namespace:
- `UInt64SingleStatsCounter` - Single global counter
- `UInt64MultiUnitStatsCounter` - Per-unit counters (e.g., per sub-partition, per SM)
- `Float32SingleStatsCounter` / `Float32MultiUnitStatsCounter` - Float versions

## Implementation Steps

### Step 1: Declare the counter

Add to an appropriate stats header. Common locations:
- `src/gpgpu-sim/mem_latency_stat.h` - Memory system stats
- `src/gpgpu-sim/shader.h` (`shader_core_stats` class) - Shader/core stats

```cpp
// Single global counter
Statistics::UInt64SingleStatsCounter my_global_counter;

// Per-unit counter (e.g., per sub-partition)
Statistics::UInt64MultiUnitStatsCounter my_per_unit_counter;
```

### Step 2: Initialize in constructor

Add to the member initializer list in the corresponding `.cc` file:

```cpp
// Single counter: (name, description, initial_value)
my_global_counter("my_global_counter", "Description of the metric", 0),

// Multi-unit counter: (name, description, num_units, initial_value)
my_per_unit_counter("my_per_unit_counter", "Per-unit description",
                    mem_config->m_n_mem_sub_partition, 0),
```

### Step 3: Increment counter in code

Place increment at the appropriate location:

```cpp
// Single counter
++m_mem_stats->my_global_counter;

// Per-unit counter (by index)
++m_mem_stats->my_per_unit_counter[m_sub_partition_id];
```

### Step 4: Register for performance counters (optional)

To make the counter visible in perf sampling output (for hardware validation), add in `gpu-sim.cc` around line 1248:

```cpp
perf_counters.add_statistics_counter(m_memory_stats->my_per_unit_counter);
```

## Reference Files

| File | Purpose |
|------|---------|
| `mem_latency_stat.h:165` | Declaration example (`chiplet_queue_full`) |
| `mem_latency_stat.cc:88-91` | Initialization example |
| `l2cache.cc:1123` | Increment example |
| `gpu-sim.cc:1248-1252` | Perf counter registration |

## Workflow

When adding a counter:
1. Ask user: counter name, type (single/multi-unit), which stats class, where to increment
2. Make the edits in order: header declaration, constructor init, increment location
3. Optionally add perf counter registration if needed for HW validation
