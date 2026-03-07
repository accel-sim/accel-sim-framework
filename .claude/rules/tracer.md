---
description: NVBit tracer tool source code
paths:
  - "util/tracer_nvbit/**"
---

# NVBit Tracer

The tracer generates SASS instruction traces from CUDA applications running on real GPU hardware using NVBit binary instrumentation.

## Directory Structure

```
util/tracer_nvbit/
├── tracer_tool/          # Main tracer implementation
│   ├── tracer_tool.cu    # NVBit tool main file
│   ├── inject_funcs.cu   # Device-side instrumentation functions
│   ├── common.h          # Shared data structures
│   └── watchdog.h        # Warpsync collective workaround
├── nvbit_release/        # NVBit SDK and example tools
├── others/               # Additional tools (BBV, occupancy, etc.)
├── run_hw_trace.py       # Python wrapper for tracing benchmarks
└── install_nvbit.sh      # Downloads NVBit SDK
```

## Source Code

### tracer_tool.cu

Main NVBit tool that instruments CUDA binaries:

**Key Callbacks:**
- `nvbit_at_init()` - Initialize tracer, parse environment variables
- `nvbit_at_cuda_event()` - Handle kernel launches, memcpy events
- `nvbit_at_function_first_load()` - Instrument each kernel's SASS code

**Environment Variables:**
| Variable | Description |
|----------|-------------|
| `DYNAMIC_KERNEL_LIMIT_START` | First kernel ID to trace |
| `DYNAMIC_KERNEL_LIMIT_END` | Last kernel ID to trace |
| `DYNAMIC_KERNEL_RANGE` | Complex range spec (e.g., `1-5,10-*,kernel_name:regex`) |
| `TRACES_FOLDER` | Output directory for traces |
| `ENABLE_COMPRESS` | Enable address compression (default: 1) |
| `LINEINFO` | Include source line info (default: 0) |
| `ALLOW_REG_VAL_TRACING` | Trace register values (version 6) |

**Global State:**
```cpp
std::map<std::string, int> opcode_to_id_map;  // SASS opcode → ID
std::unordered_map<CUcontext, int> ctx_kernelid;  // Per-context kernel counter
```

### inject_funcs.cu

Device-side instrumentation injected into every SASS instruction:

**Main Function:**
```cpp
__device__ void instrument_inst(
    int pred, int opcode_id, int32_t vpc,
    bool is_tma, uint64_t tma_param_handle, ...
)
```

Collects per-instruction:
- Active/predicate masks
- CTA/cluster IDs (SM90+)
- Memory addresses (for load/store)
- Register operands
- Immediate values

**GPU→CPU Communication:**
Uses NVBit's channel mechanism to stream `inst_trace_t` records from GPU to CPU.

### common.h

Shared data structures between host and device:

**`inst_trace_t`** - Per-instruction trace record:
```cpp
struct inst_trace_t {
    uint32_t active_mask;
    uint32_t predicate_mask;
    int cta_id_x, cta_id_y, cta_id_z;
    int cluster_cta_id_x, ...;  // SM90+ cluster info
    int opcode_id;
    uint32_t vpc;  // Virtual PC
    union {
        struct { uint64_t addrs[32]; ... } regular;  // Normal instructions
        struct { uint8_t tma_param_handle[128]; } tma;  // TMA instructions
    } inst;
};
```

### watchdog.h

Workaround for NVBit bug with `warpsync.collective` instructions.

## Trace Output Format

### Directory Structure
```
traces/
├── kernelslist.g           # List of traced kernels
├── stats.csv               # Kernel statistics
└── kernel-<id>.trace[.xz]  # Per-kernel trace files
```

### Trace File Format

**Header:**
```
-kernel name = <name>
-kernel id = <id>
-grid dim = (<gx>,<gy>,<gz>)
-block dim = (<bx>,<by>,<bz>)
-shmem = <bytes>
-nregs = <count>
-cuda stream id = <id>
-binary version = <sm_version>
-enable lineinfo = <0|1>
-trace version = <5|6>
```

**Per-Instruction:**
```
#BEGIN_TB
thread block = <x>,<y>,<z>
warp = <warp_id>
insts = <count>
<pc> <mask> <opcode> <dest> <src1,src2,...> [<addrs>|<base_stride>|<base_delta>]
...
#END_TB
```

**Address Compression Formats:**
- `0` (list_all): All 32 addresses listed
- `1` (base_stride): `base:stride` for uniform access
- `2` (base_delta): `base:d0,d1,...` for irregular access

## run_hw_trace.py

Python wrapper for batch tracing:

```bash
./run_hw_trace.py -B <benchmark_list> -D <device_num> [options]
```

**Options:**
| Flag | Description |
|------|-------------|
| `-B` | Benchmark suite(s) from define-all-apps.yml |
| `-D` | CUDA device number |
| `-l` | Limit number of kernels traced |
| `-t` | Terminate after kernel limit reached |
| `--spinlock_handling` | `none` or `fast_forward` |

## Building

```bash
./install_nvbit.sh                    # Download NVBit SDK
make -C util/tracer_nvbit/            # Build tracer
```

Output: `tracer_tool/tracer_tool.so`

## Usage

**Direct usage:**
```bash
LD_PRELOAD=./tracer_tool/tracer_tool.so ./cuda_app
```

**Via run_hw_trace.py:**
```bash
./run_hw_trace.py -B rodinia-3.1 -D 0
```

Traces saved to: `hw_run/traces/<device>/<cuda_version>/<app>/`

## Formatter

```bash
./util/tracer_nvbit/tracer_tool/format-code.sh
```

## Other Tools (others/)

| Tool | Purpose |
|------|---------|
| `bbv_tool/` | Basic block vector profiling |
| `occupancy_calc_tool/` | SM occupancy calculation |
| `silicon_checkpoint_tool/` | Checkpoint/restore support |
| `spinlock_tool/` | Spinlock detection and handling |
| `torch_hook/` | PyTorch integration for selective tracing |

## Common Modifications

### Adding New Instruction Support

1. Update `tracer_tool.cu` if special handling needed
2. Instruction info extracted automatically from NVBit

### Changing Trace Format

1. Modify `inst_trace_t` in `common.h`
2. Update writing logic in `tracer_tool.cu`
3. Update parsing in `gpu-simulator/trace-parser/trace_parser.cc`
4. Bump `TRACER_VERSION`

### Adding New Environment Variables

In `tracer_tool.cu::nvbit_at_init()`:
```cpp
const char* env = getenv("MY_VAR");
if (env) my_var = atoi(env);
```
