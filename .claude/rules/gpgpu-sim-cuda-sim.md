---
description: Functional simulation and PTX execution
paths:
  - "gpu-simulator/gpgpu-sim/src/cuda-sim/**"
---

# Functional Simulation (cuda-sim/)

PTX execution and functional correctness.

## Key Files

| File | Purpose |
|------|---------|
| `cuda-sim.h/cc` | Main functional simulator |
| `instructions.cc` | **199,736 lines** - PTX instruction semantics |
| `ptx_ir.h/cc` | PTX intermediate representation |
| `ptx_parser.h/cc` | PTX parsing and loading |
| `memory.h/cc` | Functional memory spaces |

## instructions.cc

Massive file implementing all PTX instruction semantics.

**Main entry:** `ptx_thread_info::execute_insn()` - giant switch on opcode.

### Adding a New Instruction

1. Define opcode in `opcodes.def`
2. Add case in `ptx_thread_info::execute_insn()`
3. Add timing model in `shader.cc`:
   - Update `shader_core_ctx::decode()`
   - Set execution unit type in `warp_inst_t`

## Memory Spaces

`memory.h/cc` implements functional memory:
- Global memory
- Shared memory
- Local memory
- Constant memory
- Texture memory

## Memory Coalescing

Handled in `cuda-sim.cc`:
- Memory accesses within warp combined into fewer transactions
- Rules depend on compute capability

## Reading instructions.cc

**Never read the entire file.** Use grep to find specific opcodes:
```bash
grep -n "case OP_ADD:" instructions.cc
```
