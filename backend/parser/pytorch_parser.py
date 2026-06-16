import torch
import numpy as np
from .ir import IRLayer, IRModel, TensorSpec


_OP_TYPE_MAP = {
    torch.nn.Conv2d: "Conv2d",
    torch.nn.Conv1d: "Conv1d",
    torch.nn.ConvTranspose2d: "ConvTranspose2d",
    torch.nn.BatchNorm2d: "BatchNorm2d",
    torch.nn.BatchNorm1d: "BatchNorm1d",
    torch.nn.ReLU: "ReLU",
    torch.nn.LeakyReLU: "LeakyReLU",
    torch.nn.Sigmoid: "Sigmoid",
    torch.nn.Tanh: "Tanh",
    torch.nn.MaxPool2d: "MaxPool2d",
    torch.nn.AvgPool2d: "AvgPool2d",
    torch.nn.AdaptiveAvgPool2d: "AdaptiveAvgPool2d",
    torch.nn.Linear: "Linear",
    torch.nn.Dropout: "Dropout",
    torch.nn.Flatten: "Flatten",
    torch.nn.Softmax: "Softmax",
    torch.nn.Upsample: "Upsample",
}


def parse_pytorch(file_path: str) -> IRModel:
    loaded = torch.load(file_path, map_location="cpu", weights_only=False)

    if isinstance(loaded, torch.nn.Module):
        return _parse_module(loaded)
    if isinstance(loaded, torch.jit.ScriptModule):
        return _parse_torchscript(loaded)
    if isinstance(loaded, dict):
        return _parse_checkpoint_dict(loaded)

    raise ValueError(f"Unsupported PyTorch file format: {type(loaded)}")


def _parse_module(model: torch.nn.Module) -> IRModel:
    layers = []

    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, torch.nn.Sequential):
            continue

        op_type = "Unknown"
        params = {}
        weights = {}

        for cls, op_name in _OP_TYPE_MAP.items():
            if isinstance(module, cls):
                op_type = op_name
                break

        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d)):
            k = module.kernel_size
            s = module.stride
            p = module.padding
            params = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": list(k) if isinstance(k, tuple) else [k, k],
                "stride": list(s) if isinstance(s, tuple) else [s, s],
                "padding": list(p) if isinstance(p, tuple) else [p, p],
            }
            weights["weight"] = module.weight.data.cpu().numpy()
            if module.bias is not None:
                weights["bias"] = module.bias.data.cpu().numpy()
        elif isinstance(module, torch.nn.BatchNorm2d):
            params = {
                "num_features": module.num_features,
                "eps": module.eps,
                "momentum": module.momentum,
            }
            weights["weight"] = module.weight.data.cpu().numpy()
            weights["bias"] = module.bias.data.cpu().numpy()
            weights["running_mean"] = module.running_mean.data.cpu().numpy()
            weights["running_var"] = module.running_var.data.cpu().numpy()
        elif isinstance(module, torch.nn.Linear):
            params = {
                "in_features": module.in_features,
                "out_features": module.out_features,
            }
            weights["weight"] = module.weight.data.cpu().numpy()
            if module.bias is not None:
                weights["bias"] = module.bias.data.cpu().numpy()

        layers.append(IRLayer(
            name=name,
            op_type=op_type,
            inputs=[],
            outputs=[],
            params=params,
            input_shapes=[],
            output_shapes=[],
            weights=weights,
        ))

    return IRModel(
        format="pytorch",
        producer="PyTorch",
        opset_version=None,
        layers=layers,
        inputs=[TensorSpec(name="input", shape=[1, 3, -1, -1], dtype="float32")],
        outputs=[TensorSpec(name="output", shape=[-1], dtype="float32")],
    )


def _parse_torchscript(model: torch.jit.ScriptModule) -> IRModel:
    graph = model.inlined_graph
    layers = []
    for node in graph.nodes():
        layers.append(IRLayer(
            name=node.debugName(),
            op_type=node.kind().replace("aten::", ""),
            inputs=[i.debugName() for i in node.inputs()],
            outputs=[o.debugName() for o in node.outputs()],
            params={},
            input_shapes=[],
            output_shapes=[],
            weights={},
        ))

    return IRModel(
        format="pytorch",
        producer="TorchScript",
        opset_version=None,
        layers=layers,
        inputs=[],
        outputs=[],
    )


def _parse_checkpoint_dict(ckpt: dict) -> IRModel:
    """Extract model layers from a checkpoint dict by inferring structure from weight keys.

    Handles common checkpoint formats:
    - {"model": state_dict, "optimizer": ..., "epoch": ...}
    - flat state_dict as top-level keys like "conv1.weight", "conv1.bias"
    """
    state_dict = _find_state_dict(ckpt)
    if state_dict is None:
        raise ValueError(
            "Could not find state_dict in checkpoint. Keys found: " + ", ".join(list(ckpt.keys())[:10])
        )

    layers = _state_dict_to_layers(state_dict)
    if not layers:
        raise ValueError(
            "Could not extract any layers from state_dict. "
            "The checkpoint may use nested state_dicts (e.g., {model: {encoder: {...}, decoder: {...}}}). "
            "Try saving with a flat state_dict or uploading the full model."
        )
    return IRModel(
        format="pytorch",
        producer="PyTorch Checkpoint",
        opset_version=None,
        layers=layers,
        inputs=[TensorSpec(name="input", shape=[1, 3, -1, -1], dtype="float32")],
        outputs=[TensorSpec(name="output", shape=[-1], dtype="float32")],
    )


_OPTIMIZER_KEYS = {"optimizer", "optim", "adam", "sgd", "lr_scheduler", "scheduler"}


def _has_weight_keys(d: dict) -> bool:
    """Check if dict looks like a state_dict (contains .weight/.bias keys or tensor values)."""
    if len(d) == 0:
        return False
    # Check a sample of keys (up to 5) for weight-like naming or tensor values
    samples = list(d.items())[:5]
    for k, v in samples:
        if isinstance(k, str) and (k.endswith(".weight") or k.endswith(".bias")):
            return True
        if hasattr(v, "shape"):
            return True
    return any(hasattr(v, "shape") for _, v in d.items())


def _find_state_dict(ckpt: dict) -> dict | None:
    """Try to locate the state_dict inside a checkpoint dict."""
    # Direct state_dict — top-level keys look like weight names
    if _has_weight_keys(ckpt):
        return ckpt
    # "model_state_dict" before "model" to avoid metadata dicts
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if "model" in ckpt and isinstance(ckpt["model"], dict) and _has_weight_keys(ckpt["model"]):
        return ckpt["model"]
    # Check any top-level dict values (skip optimizer/auxiliary keys)
    for key, val in ckpt.items():
        if key.lower() in _OPTIMIZER_KEYS:
            continue
        if isinstance(val, dict) and len(val) >= 1 and _has_weight_keys(val):
            return val
    return None


def _state_dict_to_layers(state_dict: dict) -> list[IRLayer]:
    """Group weight keys by layer prefix (e.g. 'features.0.weight', 'features.0.bias' → layer 'features.0')."""
    groups: dict[str, dict[str, any]] = {}
    for key, tensor in state_dict.items():
        if not hasattr(tensor, "shape"):
            continue
        # Split "layer_name.weight" → prefix="layer_name", suffix="weight"
        parts = key.rsplit(".", 1)
        prefix = parts[0] if len(parts) == 2 else ""
        suffix = parts[-1] if len(parts) == 2 else key
        if suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            if prefix not in groups:
                groups[prefix] = {}
            groups[prefix][suffix] = tensor.cpu().numpy() if hasattr(tensor, "cpu") else tensor

    layers = []
    for prefix, weights in sorted(groups.items()):
        op_type = _infer_op_type(prefix, weights)
        params = _infer_params(prefix, weights)
        layers.append(IRLayer(
            name=prefix,
            op_type=op_type,
            inputs=[],
            outputs=[],
            params=params,
            input_shapes=[],
            output_shapes=[],
            weights=weights,
        ))

    # Handle orphan keys (no .weight/.bias suffix)
    known_prefixes = set(groups.keys())
    for key, tensor in state_dict.items():
        if not hasattr(tensor, "shape"):
            continue
        parts = key.rsplit(".", 1)
        prefix = parts[0] if len(parts) == 2 else ""
        if prefix in known_prefixes:
            continue
        layers.append(IRLayer(
            name=key,
            op_type="Param",
            inputs=[],
            outputs=[],
            params={},
            input_shapes=[],
            output_shapes=[],
            weights={"value": tensor.cpu().numpy() if hasattr(tensor, "cpu") else tensor},
        ))

    return layers


def _infer_op_type(name: str, weights: dict) -> str:
    """Infer layer type from weight shapes and names."""
    w = weights.get("weight")
    if w is None:
        return "Param"

    shape = w.shape
    ndim = len(shape)
    name_lower = name.lower()

    # Named patterns
    if "fc" in name_lower or "linear" in name_lower or "dense" in name_lower:
        return "Linear"
    if "conv" in name_lower:
        if "transpose" in name_lower or "deconv" in name_lower:
            return "ConvTranspose2d"
        if ndim == 3:
            return "Conv1d"
        return "Conv2d"
    if "bn" in name_lower or "batch" in name_lower or "batchnorm" in name_lower:
        return "BatchNorm2d"
    if "ln" in name_lower or "layernorm" in name_lower or "layer_norm" in name_lower:
        return "LayerNorm"

    # Shape-based inference (no name clues)
    if ndim >= 4:
        return "Conv2d"
    if ndim == 3:
        return "Conv1d"
    if ndim == 2:
        return "Linear"
    if ndim == 1:
        return "BatchNorm2d"  # 1D weight = gamma/scale parameter
    return "Param"


def _infer_params(prefix: str, weights: dict) -> dict:
    """Infer layer params from weight shapes."""
    w = weights.get("weight")
    if w is None:
        return {}
    shape = w.shape
    ndim = len(shape)
    if ndim >= 4:
        return {
            "out_channels": int(shape[0]),
            "in_channels": int(shape[1]),
            "kernel_size": [int(shape[2]), int(shape[3])],
        }
    if ndim == 2:
        return {
            "out_features": int(shape[0]),
            "in_features": int(shape[1]),
        }
    return {}
