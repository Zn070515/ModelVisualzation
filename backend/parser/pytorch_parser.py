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

    if isinstance(loaded, dict) and "state_dict" in loaded:
        raise ValueError(
            "File contains only state_dict (weights), not full model. "
            "Save with torch.save(model, path) instead of torch.save(model.state_dict(), path)."
        )

    if isinstance(loaded, torch.nn.Module):
        return _parse_module(loaded)
    if isinstance(loaded, torch.jit.ScriptModule):
        return _parse_torchscript(loaded)
    if isinstance(loaded, dict):
        raise ValueError(
            "File contains a dict. If it's a full model checkpoint, "
            "the model architecture must be available. Try loading with the original model class."
        )

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
