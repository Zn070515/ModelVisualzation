import numpy as np
import pytest

from backend.analyzer.chain import trace_conversion_chain
from backend.analyzer.compare import compare_models
from backend.analyzer.report import generate_batch_report
from backend.parser.ir import IRLayer, IRModel


def layer(name: str, op_type: str, weights=None) -> IRLayer:
    return IRLayer(name, op_type, [], [], {}, [], [[1, 1]], weights or {})


def model(layers, fmt: str = "onnx") -> IRModel:
    return IRModel(fmt, "test", 17, layers, [], [])


# ---- Compare ----

def test_compare_models_detects_differences():
    a = model([layer("conv", "Conv2d", {"weight": np.ones((2, 1, 3, 3), dtype=np.float32)})])
    b = model([layer("conv", "Conv2d", {"weight": np.ones((4, 1, 3, 3), dtype=np.float32)}), layer("relu", "ReLU")])
    result = compare_models(a, b, "a", "b")
    assert result["summary"]["layer_count_delta"] == 1
    assert result["summary"]["param_count_delta"] == 18
    assert result["layers"][0]["status"] == "changed"


def test_compare_same_model_is_identical():
    m = model([layer("conv", "Conv2d"), layer("relu", "ReLU")])
    result = compare_models(m, m, "a", "a")
    assert result["summary"]["similarity"] == 1.0
    assert result["summary"]["changed_layers"] == 0


def test_compare_by_name_matching():
    a = model([
        layer("conv1", "Conv2d"),
        layer("conv2", "Conv2d", {"weight": np.ones((3, 3, 3, 3), dtype=np.float32)}),
    ])
    b = model([
        layer("conv2", "Conv2d", {"weight": np.ones((6, 3, 3, 3), dtype=np.float32)}),
        layer("conv1", "Conv2d"),
    ])
    result = compare_models(a, b, "a", "b")
    assert result["summary"]["changed_layers"] == 1  # conv2 params differ


# ---- Chain ----

def test_trace_conversion_chain_multiple_models():
    pt = model([layer("conv1", "Conv2d"), layer("relu1", "ReLU"), layer("dropout", "Dropout")], "pytorch")
    onnx_m = model([layer("conv1", "Conv"), layer("relu1", "Relu"), layer("shape", "Shape")], "onnx")
    tflite_m = model([layer("conv1", "CONV_2D"), layer("relu1", "RELU")], "tflite")

    result = trace_conversion_chain([pt, onnx_m, tflite_m], ["PyTorch", "ONNX", "TFLite"], ["a", "b", "c"])
    assert len(result["stages"]) == 3
    assert len(result["transitions"]) == 2
    assert result["stages"][0]["format"] == "pytorch"
    assert result["stages"][2]["format"] == "tflite"


def test_trace_chain_single_model():
    m = model([layer("relu", "ReLU")])
    result = trace_conversion_chain([m], ["PT"], ["a"])
    assert len(result["stages"]) == 1
    assert result["transitions"] == []
    assert result["summary"]["total_steps"] == 0


def test_trace_chain_detects_renames():
    src = model([layer("conv1", "Conv2d"), layer("relu1", "ReLU")], "pytorch")
    dst = model([layer("conv1", "Conv"), layer("relu1", "Relu"), layer("shape", "Shape")], "onnx")
    result = trace_conversion_chain([src, dst], ["PT", "ONNX"], ["a", "b"])
    t = result["transitions"][0]
    assert t["added_ops"] == ["Shape"] or "Shape" in t["added_ops"]
    assert any(r["from"] == "Conv2d" and r["to"] == "Conv" for r in t["renamed_ops"])


# ---- Report ----

def test_generate_batch_report_html():
    m = model([layer("fc", "Linear", {"weight": np.ones((2, 2), dtype=np.float32)})])
    result = generate_batch_report({"a": m}, "html")
    assert result["summary"]["model_count"] == 1
    assert "html" in result


def test_generate_batch_report_json():
    m = model([layer("fc", "Linear", {"weight": np.ones((3, 3), dtype=np.float32)})])
    result = generate_batch_report({"a": m}, "json")
    assert result["format"] == "json"
    assert result["summary"]["total_params"] == 9
