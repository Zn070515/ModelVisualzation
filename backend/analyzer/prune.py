from __future__ import annotations

import numpy as np


def analyze_pruning(model, model_id: str = "") -> dict:
    layers: list[dict] = []
    prunable_params = 0
    total_params = model.total_params()

    for layer in model.layers:
        weight = _find_conv_weight(layer)
        if weight is None:
            continue
        arr = np.asarray(weight, dtype=np.float32)
        channels = _channel_scores(arr)
        if not channels:
            continue
        layers.append({
            "layer_name": layer.name,
            "op_type": layer.op_type,
            "channel_importance": channels,
            "sparsity_heatmap": _sparsity_heatmap(arr),
            "total_channels": len(channels),
            "prunable_channels_30pct": int(len(channels) * 0.3),
            "prunable_channels_50pct": int(len(channels) * 0.5),
        })
        prunable_params += (
            int(np.prod(arr.shape[1:]) * int(len(channels) * 0.3))
            if arr.ndim > 1
            else int(len(channels) * 0.3)
        )

    pct = 100 * prunable_params / total_params if total_params else 0.0
    return {
        "model_id": model_id,
        "layers": layers,
        "summary": {
            "total_prunable_layers": len(layers),
            "total_prunable_params_pct": round(float(pct), 4),
            "recommended_prune_ratio": 0.3 if pct < 30 else 0.2,
        },
    }


def _channel_scores(arr: np.ndarray) -> list[dict]:
    if arr.ndim == 0:
        return []
    channels = arr.reshape((arr.shape[0], -1))
    scores: list[dict] = []
    for idx, values in enumerate(channels):
        v = values.astype(np.float64)
        l1 = float(np.mean(np.abs(v)))
        l2 = float(np.sqrt(np.mean(v ** 2)))
        fisher = float(np.mean(v ** 2))
        combined = 0.25 * l1 + 0.15 * l2 + 0.40 * fisher
        scores.append({
            "channel": int(idx),
            "l1_norm": round(l1, 8),
            "l2_norm": round(l2, 8),
            "fisher_score": round(fisher, 8),
            "activation_score": 0.0,
            "combined_importance": round(combined, 8),
            "importance": round(l1, 8),
        })
    scores.sort(key=lambda s: s["combined_importance"])
    for rank_idx, s in enumerate(scores):
        s["prune_priority"] = rank_idx + 1
    return scores


def _sparsity_heatmap(arr: np.ndarray) -> list[list[float]]:
    if arr.ndim <= 1:
        matrix = arr.reshape((1, -1))
    else:
        matrix = arr.reshape((arr.shape[0], -1))
    groups = np.array_split(matrix, min(matrix.shape[0], 16), axis=0)
    heatmap: list[list[float]] = []
    for group in groups:
        cols = np.array_split(group, min(group.shape[1], 16), axis=1)
        heatmap.append([round(float(np.mean(np.abs(col) < 1e-7)), 4) for col in cols])
    return heatmap


def _find_conv_weight(layer) -> np.ndarray | None:
    if not layer.weights:
        return None
    w = layer.weights.get("weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) >= 1:
        return w
    for val in layer.weights.values():
        if hasattr(val, "shape") and len(val.shape) >= 3:
            return val
    best = None
    best_size = 0
    for val in layer.weights.values():
        if hasattr(val, "shape") and len(val.shape) >= 1:
            size = int(np.prod(val.shape))
            if size > best_size:
                best_size = size
                best = val
    return best
