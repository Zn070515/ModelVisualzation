import torch
from backend.parser.pytorch_parser import parse_pytorch
import tempfile
import os


def build_test_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 10),
    )
    return model


def test_parse_pytorch_basic():
    model = build_test_model()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model, f.name)
        tmp = f.name
    try:
        ir = parse_pytorch(tmp)
        assert ir.format == "pytorch"
        assert len(ir.layers) >= 4
        op_types = [l.op_type for l in ir.layers]
        assert "Conv2d" in op_types
        assert "ReLU" in op_types
        assert "Linear" in op_types
        conv = next(l for l in ir.layers if l.op_type == "Conv2d")
        assert "weight" in conv.weights
        assert conv.params["in_channels"] == 3
        assert conv.params["out_channels"] == 8
    finally:
        os.unlink(tmp)
