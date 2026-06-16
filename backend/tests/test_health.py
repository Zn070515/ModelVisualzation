import numpy as np
import pytest
from backend.analyzer.health import health_check
from backend.parser.ir import IRLayer, IRModel, TensorSpec


def make_layer(name: str, op_type: str, weights: dict, params: dict = None) -> IRLayer:
    return IRLayer(
        name=name, op_type=op_type,
        inputs=[], outputs=[], params=params or {},
        input_shapes=[], output_shapes=[],
        weights=weights,
    )


def test_health_check_clean_model():
    np.random.seed(42)
    w = np.random.randn(64, 3, 3, 3).astype(np.float32) * 0.02
    b = np.random.randn(64).astype(np.float32) * 0.02
    layers = [
        make_layer("conv1", "Conv2d", {"weight": w, "bias": b}),
        make_layer("relu1", "ReLU", {}),
    ]
    model = IRModel(format="onnx", producer="test", opset_version=17, layers=layers, inputs=[], outputs=[])
    result = health_check(model)
    assert "issues" in result
    assert "summary" in result
    assert result["summary"]["critical_count"] == 0


def test_health_check_weight_outlier():
    w = np.ones((10, 10), dtype=np.float32)
    w[0, 0] = 100.0
    layers = [make_layer("fc", "Linear", {"weight": w})]
    model = IRModel(format="onnx", producer="test", opset_version=17, layers=layers, inputs=[], outputs=[])
    result = health_check(model)
    outlier_issues = [i for i in result["issues"] if i["type"] == "weight_outlier"]
    assert len(outlier_issues) > 0
    assert outlier_issues[0]["severity"] in ("warning", "critical")


def test_health_check_bn_anomaly():
    running_mean = np.ones(32, dtype=np.float32) * 10.0
    running_var = np.ones(32, dtype=np.float32) * 0.0001
    layers = [
        make_layer("bn1", "BatchNorm2d", {
            "weight": np.ones(32, dtype=np.float32),
            "bias": np.zeros(32, dtype=np.float32),
            "running_mean": running_mean,
            "running_var": running_var,
        }),
    ]
    model = IRModel(format="onnx", producer="test", opset_version=17, layers=layers, inputs=[], outputs=[])
    result = health_check(model)
    bn_issues = [i for i in result["issues"] if i["type"] == "bn_anomaly"]
    assert len(bn_issues) > 0


def test_health_check_no_layers():
    model = IRModel(format="onnx", producer="test", opset_version=17, layers=[], inputs=[], outputs=[])
    result = health_check(model)
    assert result["issues"] == []
    assert result["summary"]["total_issues"] == 0


def test_health_check_high_sparsity():
    w = np.zeros((10, 10), dtype=np.float32)
    w[0, 0] = 1.0
    layers = [make_layer("dead", "Linear", {"weight": w})]
    model = IRModel(format="onnx", producer="test", opset_version=17, layers=layers, inputs=[], outputs=[])
    result = health_check(model)
    sparsity_issues = [i for i in result["issues"] if i["type"] == "high_sparsity"]
    assert len(sparsity_issues) > 0
