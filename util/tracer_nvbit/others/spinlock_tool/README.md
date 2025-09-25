# Spinlock tool

## Description

This tool is used to detect spinlocks in the kernel code.

## Usage

```bash
# Run program first time to get the instruction histogram of the program's kernels
SPINLOCK_PHASE=0 CUDA_INJECTION64_PATH=PATH/TO/spinlock_tool.so program

# Run program second time to get another instruction histogram of the program's kernels
# At the end of nvbit, this tool will generate a file with the name of spinlock_detection/spinlock_instructions.txt
# containing the instruction indices of the spinlock instructions in the program's kernels
SPINLOCK_PHASE=1 CUDA_INJECTION64_PATH=PATH/TO/spinlock_tool.so program

# To fast forward the spinlock instructions with accel-sim tracer, you can use the following command
ENABLE_SPINLOCK_FAST_FORWARD=1 CUDA_INJECTION64_PATH=PATH/TO/tracer_tool.so program
```
