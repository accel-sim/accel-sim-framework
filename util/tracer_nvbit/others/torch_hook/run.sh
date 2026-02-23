#!/bin/bash
# Wrapper script for PyTorch NVBit tracing
# Usage: ./run.sh <command> [args...]
# Example: ./run.sh python3 vllm_example.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_TOOL_DIR="$(cd "$SCRIPT_DIR/../../tracer_tool" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 python3 vllm_example.py"
    exit 1
fi

# Set up tracing environment
export PYTHONPATH="${TRACER_TOOL_DIR}:$PYTHONPATH"
export CUDA_INJECTION64_PATH="${TRACER_TOOL_DIR}/tracer_tool.so"

# vLLM-specific settings
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Disable NVBit instrumentation by default; enable in your script if needed
export NVBIT_INSTRUMENTATION_ENABLED=0

# spinlock handling
export ENABLE_SPINLOCK_FAST_FORWARD=1
export SPINLOCK_ITER_TO_KEEP=5

echo "Environment configured for PyTorch NVBit tracing:"
echo "  PYTHONPATH includes: ${TRACER_TOOL_DIR}"
echo "  CUDA_INJECTION64_PATH: ${CUDA_INJECTION64_PATH}"
echo "Running: $@"
echo ""

exec "$@"
