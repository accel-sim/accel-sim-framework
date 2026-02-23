from vllm import LLM, SamplingParams
from torch_hook import TorchModelHookWrapper, hook_nvtx, hook_nvbit_to_layer
import torch

prompts = [
    "Hello, my name is"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m", enforce_eager=True)

def print_layer_names(model: torch.nn.Module):
    print("Available layers:")
    for name, module in model.named_modules():
        print(f"  {name}: {module.__class__.__name__}")

def apply_nvbit_hook(model: torch.nn.Module, layers_to_trace: list[str]):
    hook_wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        hook_nvbit_to_layer(hook_wrapper, layer)


llm.collective_rpc(lambda self: print_layer_names(self.model_runner.model))

layers_to_trace = ["model.decoder.layers.10.self_attn"]
llm.collective_rpc(lambda self: apply_nvbit_hook(self.model_runner.model, layers_to_trace))
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm