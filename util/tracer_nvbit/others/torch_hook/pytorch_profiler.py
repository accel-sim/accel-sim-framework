from ctypes import CDLL, c_char_p
from typing import Callable, Optional, Any, Union
import logging
import torch
import nvtx
import os

# Types
PytorchPreHook = Callable[[torch.nn.Module, torch.Tensor], Optional[torch.Tensor]]
PytorchPostHook = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]

# Generic hook wrapper for torch models
class TorchModelHookWrapper:
    def __init__(self, model: torch.nn.Module, top_name: str="top_model"):
        self.model = model
        for name, module in self.model.named_modules():
            module.name = name
        model.name = top_name

    def register_module_forward_pre_hook(self, hook: PytorchPreHook):
        logging.debug(f"Registering forward pre hook for all modules: {hook}")
        return torch.nn.modules.module.register_module_forward_pre_hook(hook)

    def register_module_forward_hook(self, hook: PytorchPostHook):
        logging.debug(f"Registering forward hook for all modules: {hook}")
        return torch.nn.modules.module.register_module_forward_hook(hook)
    
    def register_forward_pre_hook_by_name(self, module_name: str, hook: PytorchPreHook):
        logging.debug(f"Registering forward pre hook for module {module_name}: {hook}")
        for name, module in self.model.named_modules():
            if name == module_name:
                return module.register_forward_pre_hook(hook)
        raise ValueError(f"Module with name {module_name} not found in model")
    
    def register_forward_hook_by_name(self, module_name: str, hook: PytorchPostHook):
        logging.debug(f"Registering forward hook for module {module_name}: {hook}")
        for name, module in self.model.named_modules():
            if name == module_name:
                return module.register_forward_hook(hook)
        raise ValueError(f"Module with name {module_name} not found in model")

# Hooking NVTX
def convert_hook_operand_to_str(input: Union[torch.Tensor, tuple, Any]):
    # If the element in the tuple is a tensor, print the shape, else print the type
    if isinstance(input, tuple):
        return ", ".join([convert_hook_operand_to_str(x) for x in input])
    elif isinstance(input, torch.Tensor):
        return f"Tensor shape {input.shape}"
    else:
        return f"type {type(input)}"

# Use nvtx to mark the start and end of each layer
def nvtx_mark_start(module: torch.nn.Module, input):
    if (not hasattr(module, "name")):
        logging.warning(f"Module {module.__class__.__name__} has no instance name, use class name instead")
        module.name = module.__class__.__name__

    logging.debug(f"nvtx_mark_start: {module.name} with input {convert_hook_operand_to_str(input)}")
    nvtx.push_range(module.name)

def nvtx_mark_end(module: torch.nn.Module, input, output):
    logging.debug(f"nvtx_mark_end: {module.name} with input {convert_hook_operand_to_str(input)} and output {convert_hook_operand_to_str(output)}")
    nvtx.pop_range()

def hook_nvtx(hook_wrapper: TorchModelHookWrapper):
    hook_wrapper.register_module_forward_pre_hook(nvtx_mark_start)
    hook_wrapper.register_module_forward_hook(nvtx_mark_end)

# Hooking NVBit
# Global states for NVBit tracer
g_found_nvbit_tracer = False
g_nvbit_tracer_try_loading = True
g_nvbit_tracer = None
def hook_nvbit_to_layer(hook_wrapper: TorchModelHookWrapper, layer_for_tracing: str):
    global g_found_nvbit_tracer, g_nvbit_tracer, g_nvbit_tracer_try_loading
    
    # First time initialization for nvbit tracer callbacks
    if g_nvbit_tracer_try_loading and os.environ.get("CUDA_INJECTION64_PATH"):
        logging.info(f"CUDA_INJECTION64_PATH: {os.environ['CUDA_INJECTION64_PATH']}")
        g_nvbit_tracer = CDLL(os.environ["CUDA_INJECTION64_PATH"])
        # Set the return type to None to match void return type
        # These two knobs will determine if the tracing should be enabled or disabled
        # The third function is to set the tag for the tracing, used as part of the
        # trace file name
        try:    
            g_nvbit_tracer.enable_nvbit_instrumentation.restype = None
            g_nvbit_tracer.disable_nvbit_instrumentation.restype = None
            g_nvbit_tracer.set_nvbit_instrumentation_tag.restype = None
            g_nvbit_tracer.set_nvbit_instrumentation_tag.argtypes = [c_char_p]
            g_found_nvbit_tracer = True
        except AttributeError:
            logging.info("NVBit tracer not found, ignoring...")
            g_found_nvbit_tracer = False

        # Just try loading once
        g_nvbit_tracer_try_loading = False
        

    # Select the layer for nvbit tracing with hook function by name
    if g_found_nvbit_tracer:
        logging.info(f"Found nvbit tracer, attaching hooks to layer {layer_for_tracing}")
        hook_wrapper.register_forward_pre_hook_by_name(layer_for_tracing, lambda module, input: g_nvbit_tracer.enable_nvbit_instrumentation())
        hook_wrapper.register_forward_hook_by_name(layer_for_tracing, lambda module, input, output: g_nvbit_tracer.disable_nvbit_instrumentation())
        hook_wrapper.register_forward_pre_hook_by_name(layer_for_tracing, lambda module, input: g_nvbit_tracer.set_nvbit_instrumentation_tag(c_char_p(layer_for_tracing.encode('ascii'))))