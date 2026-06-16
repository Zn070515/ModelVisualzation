import numpy as np

from .profile import _estimate_flops


HARDWARE_OPS_BENCH = {
    "cpu": {"conv": 2.5e5, "linear": 6.0e5, "elementwise": 1.5e6},
    "gpu": {"conv": 2.2e6, "linear": 3.0e6, "elementwise": 6.0e6},
    "edge_tpu": {"conv": 1.3e6, "linear": 9.0e5, "elementwise": 2.5e6},
}


def estimate_perf(model, hardware: str = "cpu", model_id: str = "") -> dict:
    if hardware not in HARDWARE_OPS_BENCH:
        raise ValueError(f"Unsupported hardware: {hardware}")

    bench = HARDWARE_OPS_BENCH[hardware]
    layers = []
    for layer in model.layers:
        flops = _estimate_flops(layer)
        params = layer.param_count()
        op_class = _op_class(layer.op_type)
        throughput = bench[op_class]
        latency_us = (flops / throughput) if flops else _shape_work(layer) / throughput
        memory_read = params * 4
        memory_write = _output_bytes(layer)
        if hardware == "edge_tpu" and op_class not in {"conv", "elementwise"}:
            latency_us *= 1.8
        layers.append({
            "name": layer.name,
            "op_type": layer.op_type,
            "params": params,
            "flops": int(flops),
            "est_latency_us": round(float(latency_us), 3),
            "is_bottleneck": False,
            "bottleneck_score": 0.0,
            "memory_read_bytes": int(memory_read),
            "memory_write_bytes": int(memory_write),
        })

    total = sum(layer["est_latency_us"] for layer in layers)
    threshold = max(total * 0.2, 1e-9)
    bottlenecks = []
    for layer in layers:
        score = layer["est_latency_us"] / total if total else 0
        layer["bottleneck_score"] = round(score, 4)
        layer["is_bottleneck"] = layer["est_latency_us"] >= threshold and total > 0
        if layer["is_bottleneck"]:
            bottlenecks.append(layer["name"])

    return {
        "model_id": model_id,
        "hardware": hardware,
        "layers": layers,
        "summary": {
            "total_latency_us": round(float(total), 3),
            "total_latency_ms": round(float(total) / 1000, 3),
            "bottleneck_layers": bottlenecks,
            "bottleneck_count": len(bottlenecks),
            "memory_total_bytes": int(sum(layer["memory_read_bytes"] + layer["memory_write_bytes"] for layer in layers)),
        },
    }


def _op_class(op_type: str) -> str:
    lower = op_type.lower()
    if "conv" in lower:
        return "conv"
    if "linear" in lower or "gemm" in lower or "fully" in lower:
        return "linear"
    return "elementwise"


def _shape_work(layer) -> int:
    if layer.output_shapes:
        return int(np.prod([dim for dim in layer.output_shapes[0] if dim and dim > 0]))
    return max(layer.param_count(), 1)


def _output_bytes(layer) -> int:
    if not layer.output_shapes:
        return 0
    return int(np.prod([dim for dim in layer.output_shapes[0] if dim and dim > 0]) * 4)
