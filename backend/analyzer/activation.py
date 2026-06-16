import io

import numpy as np


def collect_activations(model, sample_bytes: bytes, model_id: str = "", layer_names: list[str] | None = None) -> dict:
    sample = _load_sample(sample_bytes)
    selected = set(layer_names or [])
    activations = []

    current = sample
    for layer in model.layers:
        if selected and layer.name not in selected:
            continue
        current = _synthetic_forward(layer, current)
        stats = _stats(current)
        dead = _dead_indices(current)
        activations.append({
            "layer_name": layer.name,
            "output_shape": [int(dim) for dim in current.shape],
            "stats": stats,
            "dead_neurons_pct": round(100 * len(dead) / max(_channel_count(current), 1), 4),
            "dead_neuron_indices": dead[:100],
            "saturation_pct": round(float(np.mean(np.abs(current) > 6.0) * 100), 4),
        })

    layers_with_dead = sum(1 for item in activations if item["dead_neurons_pct"] > 0)
    return {
        "model_id": model_id,
        "activations": activations,
        "summary": {
            "total_layers_analyzed": len(activations),
            "layers_with_dead_neurons": layers_with_dead,
            "overall_dead_neuron_pct": round(float(np.mean([item["dead_neurons_pct"] for item in activations])) if activations else 0.0, 4),
        },
    }


def _load_sample(sample_bytes: bytes) -> np.ndarray:
    try:
        arr = np.load(io.BytesIO(sample_bytes), allow_pickle=False)
    except Exception:
        arr = np.frombuffer(sample_bytes, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((1, 1), dtype=np.float32)
    return arr


def _synthetic_forward(layer, arr: np.ndarray) -> np.ndarray:
    weight_scale = 1.0
    if layer.weights:
        values = [np.asarray(w, dtype=np.float32) for w in layer.weights.values() if hasattr(w, "shape")]
        if values:
            weight_scale = float(np.mean([np.std(value) for value in values])) or 1.0
    out = arr * weight_scale
    lower = layer.op_type.lower()
    if "relu" in lower:
        out = np.maximum(out, 0)
    elif "sigmoid" in lower:
        out = 1 / (1 + np.exp(-np.clip(out, -20, 20)))
    elif "tanh" in lower:
        out = np.tanh(out)
    return out.astype(np.float32)


def _stats(arr: np.ndarray) -> dict:
    hist = np.histogram(arr.ravel(), bins=50)
    return {
        "mean": round(float(np.mean(arr)), 8),
        "std": round(float(np.std(arr)), 8),
        "min": round(float(np.min(arr)), 8),
        "max": round(float(np.max(arr)), 8),
        "sparsity": round(float(np.mean(np.abs(arr) < 1e-7)), 6),
        "histogram": {
            "counts": hist[0].astype(int).tolist(),
            "bin_edges": [round(float(edge), 8) for edge in hist[1].tolist()],
        },
    }


def _channel_count(arr: np.ndarray) -> int:
    return int(arr.shape[1]) if arr.ndim >= 2 else int(arr.shape[0])


def _dead_indices(arr: np.ndarray) -> list[int]:
    if arr.ndim < 2:
        return [int(i) for i, value in enumerate(arr.ravel()) if abs(float(value)) < 1e-7]
    axes = tuple(index for index in range(arr.ndim) if index != 1)
    channel_mean = np.mean(np.abs(arr), axis=axes)
    return [int(index) for index, value in enumerate(channel_mean) if float(value) < 1e-7]
