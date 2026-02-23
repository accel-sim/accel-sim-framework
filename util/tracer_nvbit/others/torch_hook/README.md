# PyTorch NVBit Tracer Hook

A PyTorch hook module for tracing CUDA kernels using NVBit instrumentation. This tool allows you to selectively trace specific layers of a PyTorch model for GPU simulation with Accel-Sim.

## Overview

This module provides:
- **TorchModelHookWrapper**: A wrapper class to register forward hooks on PyTorch models
- **hook_nvtx**: NVTX range markers for profiling with Nsight Systems
- **hook_nvbit_to_layer**: NVBit instrumentation hooks for tracing specific layers

## Quick Start

Use the `run.sh` wrapper script to run your Python script with all required environment variables:

```bash
./run.sh python3 vllm_example.py
```

The wrapper automatically sets up the tracing environment and executes your command.

## Environment Variables

The `run.sh` wrapper sets the following environment variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `PYTHONPATH` | tracer_tool directory | Enables importing from torch_hook |
| `CUDA_INJECTION64_PATH` | tracer_tool.so | Path to the NVBit tracer shared library |
| `NVBIT_INSTRUMENTATION_ENABLED` | `0` | Disabled by default; enable in your script |
| `ENABLE_SPINLOCK_FAST_FORWARD` | `1` | Enables spinlock fast-forwarding |
| `SPINLOCK_ITER_TO_KEEP` | `5` | Number of spinlock iterations to keep |

### vLLM-Specific Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `1` | Allow insecure serialization |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | Disable multiprocessing (required for tracing) |

## Usage

### vLLM Example

Run the included example:

```bash
./run.sh python3 vllm_example.py
```

In your Python code:

```python
from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer

def apply_nvbit_hook(model, layers_to_trace):
    hook_wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        hook_nvbit_to_layer(hook_wrapper, layer)

# Apply hooks via vLLM's collective_rpc
layers_to_trace = ["model.decoder.layers.10.self_attn"]
llm.collective_rpc(lambda self: apply_nvbit_hook(self.model_runner.model, layers_to_trace))
```

### Generic PyTorch Usage

For non-vLLM models with direct access:

```bash
./run.sh python3 your_script.py
```

```python
from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_nvtx

model = YourModel()
hook_wrapper = TorchModelHookWrapper(model)

hook_nvbit_to_layer(hook_wrapper, "layer_name_to_trace")
```

## API Reference

### TorchModelHookWrapper

```python
TorchModelHookWrapper(model: torch.nn.Module, top_name: str = "top_model")
```

Methods:
- `register_module_forward_pre_hook(hook)`: Register a pre-forward hook for all modules
- `register_module_forward_hook(hook)`: Register a post-forward hook for all modules
- `register_forward_pre_hook_by_name(module_name, hook)`: Register a pre-forward hook for a specific module
- `register_forward_hook_by_name(module_name, hook)`: Register a post-forward hook for a specific module

### hook_nvtx

```python
hook_nvtx(hook_wrapper: TorchModelHookWrapper)
```

Adds NVTX push/pop range markers around every module's forward pass for profiling.

### hook_nvbit_to_layer

```python
hook_nvbit_to_layer(hook_wrapper: TorchModelHookWrapper, layer_for_tracing: str)
```

Enables NVBit instrumentation only during the forward pass of the specified layer. The trace output will be tagged with the layer name.

## Debug Logging

Enable debug logging to see hook activity:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
