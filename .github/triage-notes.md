# About this repository (for the issue-triage model)

Accel-Sim is a GPU simulation framework. Two source trees are searched:

- `accel-sim` — the framework: the trace-driven front end, the NVBit tracer,
  job launching, configs, plotting.
- `gpgpu-sim` — the GPGPU-Sim performance model, cloned at build time (it is
  not a submodule). Most questions about the simulator core resolve here.

| Topic | Where |
| --- | --- |
| SM, warp scheduler, caches, memory partition, interconnect, mbarriers, power | `gpgpu-sim:src/` |
| `gpgpusim.config` options and their parsing | `gpgpu-sim:src/gpgpu-sim/gpu-sim.cc`, `gpgpu-sim:configs/tested-cfgs/` |
| Trace parsing, ISA decode, trace-driven front end | `accel-sim:gpu-simulator/` (`trace-parser/`, `trace-driven/`, `ISA_Def/`) |
| `trace.config` options | `accel-sim:gpu-simulator/configs/tested-cfgs/` |
| NVBit tracer, trace generation | `accel-sim:util/tracer_nvbit/` |
| Job launching, benchmark and config YAML, stat collection | `accel-sim:util/job_launching/` |
| HW-vs-sim correlation, plotting | `accel-sim:util/plotting/` |

Config options are registered with `option_parser_register(...)`; quote that
default, not the help string's.

Releases: Accel-Sim 2.0 (Aug 2026) rewrote large parts of the simulator and
tracer (Hopper support, warp groups, `mbarrier` synchronization, the chiplet
memory subsystem, a new trace format). Past issues of era 1.x predate it; the
trees searched are 2.x. If the reporter is clearly on 1.x, say which version
the answer was checked against.

Labels:
- `bug` — unexpected behaviour, crash, assertion failure
- `build` — compile, link or CMake failure, toolchain
- `question` — asks how something works
- `tracer` — the NVBit tracer (`util/tracer_nvbit/`)
- `simulator` — the simulator core (gpgpu-sim)
- `correlation` — HW vs sim accuracy, `plot-correlation.py`
- `config` — simulator or trace config files
- `docs` — missing or wrong documentation
- `needs-repro` — not enough information to reproduce
- `good-first-issue` — small, well scoped
