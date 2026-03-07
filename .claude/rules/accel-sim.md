---
description: Accel-Sim framework documentation
paths:
  - "**"
---

# Accel-Sim Framework

This file provides guidance for working with code in this repository.

## Repository Structure

This repository contains **two separate git repositories**:

1. **Accel-Sim Framework** (root directory): The main framework for GPU simulation and validation
2. **GPGPU-Sim** (nested at `gpu-simulator/gpgpu-sim/`): A submodule that provides the core performance model

When making changes or commits, be aware of which repository you are working in. The `gpu-simulator/gpgpu-sim/` directory has its own `.git` and should be treated as a separate project.

## Build System

```bash
# Setup environment first
source ./gpu-simulator/setup_environment.sh
# or enable debug: source ./gpu-simulator/setup_environment.sh debug

# Configure and build
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build/<release-or-debug>
cmake --build ./gpu-simulator/build/<release-or-debug> -j8
cmake --install ./gpu-simulator/build/<release-or-debug>
```

The executable will be in: `./gpu-simulator/bin/release/accel-sim.out`

Build artifacts are placed in `./gpu-simulator/build/{release,debug}/`

## Formatter

Run before committing:
```bash
./gpu-simulator/format-code.sh                    # Accel-Sim wrapper
./util/tracer_nvbit/tracer_tool/format-code.sh    # Tracer
./gpu-simulator/gpgpu-sim/format-code.sh          # GPGPU-Sim (separate repo)
```

## Key Components

### Tracer (util/tracer_nvbit/)
Generates SASS traces from CUDA applications running on real hardware using NVBit:
```bash
export CUDA_INSTALL_PATH=<cuda_path>
./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/
./util/tracer_nvbit/run_hw_trace.py -B <benchmark> -D <device-num>
```

Traces are saved to `./hw_run/traces/`

### Simulation Runner (util/job_launching/)
The workload launch manager that handles running simulations:
```bash
# SASS trace-driven mode
./util/job_launching/run_simulations.py -B <benchmark> -C <config> -T <trace-path> -N <run-name>

# PTX execution-driven mode
./util/job_launching/run_simulations.py -B <benchmark> -C <config> -N <run-name>

# Monitor progress
./util/job_launching/monitor_func_test.py -v -N <run-name>

# Collect statistics
./util/job_launching/get_stats.py -N <run-name> | tee stats.csv
```

### Configuration System
- Simulator configs: `./gpu-simulator/gpgpu-sim/configs/tested-cfgs/`
- Trace configs: `./gpu-simulator/configs/tested-cfgs/`
- Standard configs defined in: `./util/job_launching/configs/define-standard-cfgs.yml`
- Benchmark definitions: `./util/job_launching/apps/define-all-apps.yml`

## Simulation Modes

Accel-Sim supports two execution modes:

1. **SASS Trace-Driven**: Replays traces collected from real GPU hardware (requires traces)
2. **PTX Execution-Driven**: Functionally simulates PTX code without traces

## Running Directory Structure

Simulations run in: `./sim_run_<cuda-version>/<app_name>/<app_args>/<config>/`

Each directory contains:
- Config files copied from tested-cfgs
- Symbolic links to data files and traces
- Job output files (`.o<jobId>` and `.e<jobId>`)
- `torque.sim` or similar script to reproduce the run

To debug a specific simulation:
```bash
cd sim_run_*/app/args/config/
gdb --args $(cat justrun.sh)
```

## AccelWattch Power Model

Enable power modeling by setting in the config file:
```
-power_simulation_enabled 1
-power_simulation_mode 0  # 0=SASS_SIM/PTX_SIM, 1=HW, 2=HYBRID
-accelwattch_xml_file <filename>.xml
```

Power reports are saved to `accelwattch_power_report.log` in the simulation directory.

## Important Environment Variables

- `CUDA_INSTALL_PATH`: Path to CUDA toolkit (required)
- `ACCELSIM_ROOT`: Set by setup_environment.sh to framework root
- `GPGPUSIM_ROOT`: Set to gpgpu-sim directory location
- `ACCELSIM_CONFIG`: Build type (release or debug)