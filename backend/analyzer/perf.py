import numpy as np

from .profile import _estimate_flops

HARDWARE_PROFILES = {
    "i9_13900k": {
        "label": "Core i9-13900K",
        "conv_ops": 1.2e6,
        "linear_ops": 2.8e6,
        "elem_ops": 8.0e6,
        "mem_bw_gb_s": 89.6,
        "type": "cpu",
    },
    "rtx_4090": {
        "label": "RTX 4090",
        "conv_ops": 8.2e7,
        "linear_ops": 1.3e8,
        "elem_ops": 3.0e8,
        "mem_bw_gb_s": 1008.0,
        "type": "gpu",
    },
    "apple_m2": {
        "label": "Apple M2",
        "conv_ops": 5.0e6,
        "linear_ops": 1.0e7,
        "elem_ops": 3.0e7,
        "mem_bw_gb_s": 100.0,
        "type": "cpu",
    },
    "rpi4": {
        "label": "Raspberry Pi 4",
        "conv_ops": 1.5e4,
        "linear_ops": 5.0e4,
        "elem_ops": 2.0e5,
        "mem_bw_gb_s": 4.4,
        "type": "edge",
    },
}


_LEGACY_HW_MAP = {
    "cpu": "i9_13900k",
    "gpu": "rtx_4090",
    "edge_tpu": "rpi4",
}


def estimate_perf(model, hardware: str = "i9_13900k", model_id: str = "") -> dict:
    hardware = _LEGACY_HW_MAP.get(hardware, hardware)
    if hardware not in HARDWARE_PROFILES:
        raise ValueError(f"Unsupported hardware: {hardware}. Choose from: {list(HARDWARE_PROFILES.keys())}")

    profile = HARDWARE_PROFILES[hardware]
    mem_bw_bytes_s = profile["mem_bw_gb_s"] * 1e9
    layers = []

    for layer in model.layers:
        flops = _estimate_flops(layer)
        params = layer.param_count()
        op_class = _op_class(layer.op_type)
        throughput = profile.get(f"{op_class}_ops", profile["elem_ops"])
        compute_latency_us = (flops / throughput) * 1e6 if flops else 0
        memory_read_bytes = params * 4
        memory_write_bytes = _output_bytes(layer)
        total_memory_bytes = memory_read_bytes + memory_write_bytes
        memory_latency_us = total_memory_bytes / mem_bw_bytes_s * 1e6
        est_latency_us = compute_latency_us + memory_latency_us

        if compute_latency_us > memory_latency_us * 1.5:
            bound = "compute"
        elif memory_latency_us > compute_latency_us * 1.5:
            bound = "memory"
        else:
            bound = "balanced"

        layers.append({
            "name": layer.name,
            "op_type": layer.op_type,
            "params": params,
            "flops": int(flops),
            "est_latency_us": round(float(est_latency_us), 3),
            "is_bottleneck": False,
            "bottleneck_score": 0.0,
            "memory_read_bytes": int(memory_read_bytes),
            "memory_write_bytes": int(memory_write_bytes),
            "compute_latency_us": round(float(compute_latency_us), 3),
            "memory_latency_us": round(float(memory_latency_us), 3),
            "bound": bound,
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
    return "elem"


def _output_bytes(layer) -> int:
    if not layer.output_shapes:
        return 0
    return int(np.prod([dim for dim in layer.output_shapes[0] if dim and dim > 0]) * 4)
