"""Shared BF16 model loading and multi-GPU resource configuration."""

from __future__ import annotations


def load_kwargs(config: dict) -> dict:
    import torch

    memory = {
        int(key) if str(key).isdigit() else key: value
        for key, value in config.get("max_memory", {}).items()
    }
    kwargs = {
        "dtype": torch.bfloat16,
        "local_files_only": True,
        "attn_implementation": config["attn_implementation"],
        "device_map": config["device_map"],
    }
    if memory:
        kwargs["max_memory"] = memory
    return kwargs


def input_device(model):
    return model.get_input_embeddings().weight.device


def move_inputs(inputs, device):
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }


def _linear_patch_forward(module, hidden_states):
    import torch.nn.functional as functional

    weight = module.proj.weight
    flat = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
    output = functional.linear(flat, weight.reshape(weight.shape[0], -1), module.proj.bias)
    return output.view(-1, module.embed_dim)


def linearize_patch_embed(model):
    """Rewrite single-patch non-overlapping Conv3d into an equivalent linear projection, matching the official d3 runtime."""
    import types
    import torch

    count = 0
    for module in model.modules():
        projection = getattr(module, "proj", None)
        if (
            isinstance(projection, torch.nn.Conv3d)
            and tuple(projection.kernel_size) == tuple(projection.stride)
            and hasattr(module, "embed_dim")
        ):
            module.forward = types.MethodType(_linear_patch_forward, module)
            count += 1
    return count
