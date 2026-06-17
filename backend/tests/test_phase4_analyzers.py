import numpy as np

from backend.analyzer.activation import collect_activations
from backend.analyzer.perf import estimate_perf
from backend.analyzer.prune import analyze_pruning
from backend.analyzer.quant import simulate_quantization
from backend.parser.ir import IRLayer, IRModel


def make_model() -> IRModel:
    layers = [
        IRLayer("conv", "Conv2d", [], [], {}, [[1, 3, 8, 8]], [[1, 4, 6, 6]], {"weight": np.random.randn(4, 3, 3, 3).astype(np.float32)}),
        IRLayer("relu", "ReLU", [], [], {}, [[1, 4, 6, 6]], [[1, 4, 6, 6]], {}),
    ]
    return IRModel("onnx", "test", 17, layers, [], [])


def test_estimate_perf():
    result = estimate_perf(make_model(), "i9_13900k", "m")
    assert result["summary"]["total_latency_us"] > 0
    assert result["layers"][0]["flops"] > 0


def test_simulate_quantization():
    result = simulate_quantization(make_model(), "m", 8)
    assert result["bits"] == 8
    assert result["summary"]["worst_layer"] == "conv"


def test_collect_activations():
    sample = np.ones((1, 4, 6, 6), dtype=np.float32)
    result = collect_activations(make_model(), sample.tobytes(), "m")
    assert result["summary"]["total_layers_analyzed"] == 2
    assert "histogram" in result["activations"][0]["stats"]


def test_analyze_pruning():
    result = analyze_pruning(make_model(), "m")
    assert result["summary"]["total_prunable_layers"] == 1
    assert result["layers"][0]["total_channels"] == 4
