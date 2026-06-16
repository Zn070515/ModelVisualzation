import numpy as np
import pytest
from backend.analyzer.weights import compute_layer_weight_stats, compute_weight_overview
from backend.parser.ir import IRLayer, IRModel, TensorSpec


def make_layer(name: str, op_type: str, weights: dict) -> IRLayer:
    return IRLayer(
        name=name, op_type=op_type,
        inputs=[], outputs=[], params={},
        input_shapes=[], output_shapes=[],
        weights=weights,
    )


def test_compute_layer_weight_stats_basic():
    w = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    layer = make_layer("fc", "Linear", {"weight": w})
    stats = compute_layer_weight_stats(layer)

    assert stats["layer_name"] == "fc"
    assert stats["op_type"] == "Linear"
    assert "weight" in stats["weights"]
    ws = stats["weights"]["weight"]
    assert ws["shape"] == [2, 2]
    assert ws["mean"] == pytest.approx(2.5)
    assert ws["min"] == 1.0
    assert ws["max"] == 4.0
    assert "histogram" in ws
    assert len(ws["histogram"]["counts"]) == 50
    assert len(ws["histogram"]["bin_edges"]) == 51


def test_compute_layer_weight_stats_sparsity():
    w = np.array([[0.0, 0.0], [0.0, 1e-6]], dtype=np.float32)
    layer = make_layer("sparse", "Conv2d", {"weight": w})
    stats = compute_layer_weight_stats(layer)
    ws = stats["weights"]["weight"]
    assert ws["sparsity"] == pytest.approx(0.75)


def test_compute_layer_weight_stats_multiple_weights():
    w = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([0.5], dtype=np.float32)
    layer = make_layer("fc2", "Linear", {"weight": w, "bias": b})
    stats = compute_layer_weight_stats(layer)
    assert "weight" in stats["weights"]
    assert "bias" in stats["weights"]
    assert stats["weights"]["bias"]["mean"] == 0.5


def test_compute_weight_overview():
    w1 = np.array([[1.0, 2.0]], dtype=np.float32)
    w2 = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    layers = [
        make_layer("conv1", "Conv2d", {"weight": w1}),
        make_layer("fc1", "Linear", {"weight": w2}),
    ]
    model = IRModel(
        format="onnx", producer="test", opset_version=17,
        layers=layers, inputs=[], outputs=[],
    )
    overview = compute_weight_overview(model)

    assert len(overview["layers"]) == 2
    assert overview["layers"][0]["param_count"] == 2
    assert overview["layers"][1]["param_count"] == 3
    assert overview["global_stats"]["total_weight_params"] == 5
    assert "overall_mean" in overview["global_stats"]


def test_compute_weight_overview_empty():
    model = IRModel(
        format="onnx", producer="test", opset_version=17,
        layers=[], inputs=[], outputs=[],
    )
    overview = compute_weight_overview(model)
    assert overview["layers"] == []
    assert overview["global_stats"]["total_weight_params"] == 0
