#!/bin/bash
# Wrapper to run any Python / CUDA workload under the Accel-Sim NVBit tracer.
# Generalized: pass any command to run (e.g. a vLLM, PyTorch, or diffusion script);
# this only sets up the tracing environment and execs it.
#
# Usage:   ./run.sh <command> [args...]
# Example: ./run.sh python3 vllm_example.py
#          ./run.sh python3 sdxl_inference_trace.py --num_steps 20 --image_size 1024

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# util/tracer_nvbit  (this script lives in others/torch_hook/)
TRACER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 python3 vllm_example.py"
    exit 1
fi

# --- NVBit tracer ---
export PYTHONPATH="${TRACER_DIR}/others:${TRACER_DIR}/tracer_tool:${PYTHONPATH}"
export CUDA_INJECTION64_PATH="${TRACER_DIR}/tracer_tool/tracer_tool.so"
# Off by default; enable per-layer from your script (e.g. hook_nvbit_to_layer).
export NVBIT_INSTRUMENTATION_ENABLED=0
# Trace register values (tensor descriptors, mbarrier operands).
export ALLOW_REG_VAL_TRACING=1

# --- Spinlock handling: mark_region (0=none, 1=fast_forward, 2=mark_region) ---
export SPINLOCK_HANDLING_MODE=2

# --- vLLM-specific (harmless for non-vLLM workloads) ---
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "=== Accel-Sim NVBit tracing ==="
echo "  CUDA_INJECTION64_PATH: ${CUDA_INJECTION64_PATH}"
echo "  SPINLOCK_HANDLING_MODE: ${SPINLOCK_HANDLING_MODE} (mark_region)"
echo "  TRACES_FOLDER:          ${TRACES_FOLDER:-<unset, defaults to ./traces>}"
echo "  Running:                $*"
echo ""

exec "$@"
