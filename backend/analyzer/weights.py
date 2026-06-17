from __future__ import annotations

import numpy as np


def compute_layer_weight_stats(layer) -> dict:
    weight_stats: dict[str, dict] = {}
    for wname, warr in layer.weights.items():
        if not hasattr(warr, "shape"):
            continue
        arr = np.asarray(warr, dtype=np.float32).ravel()
        hist = np.histogram(arr, bins=50)
        weight_stats[wname] = {
            "shape": [int(d) for d in warr.shape],
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sparsity": float(np.sum(np.abs(arr) < 1e-7) / arr.size),
            "histogram": {
                "counts": hist[0].tolist(),
                "bin_edges": [round(float(e), 6) for e in hist[1]],
            },
        }
    return {"layer_name": layer.name, "op_type": layer.op_type, "weights": weight_stats}


def compute_weight_overview(ir_model) -> dict:
    layer_summaries: list[dict] = []
    all_means: list[float] = []
    all_stds: list[float] = []
    all_sparsities: list[float] = []
    total_wp = 0

    for layer in ir_model.layers:
        pcount = layer.param_count()
        if pcount == 0 or not layer.weights:
            continue
        layer_mean_vals: list[float] = []
        layer_std_vals: list[float] = []
        layer_sparsity_vals: list[float] = []
        for w in layer.weights.values():
            if not hasattr(w, "shape"):
                continue
            arr = np.asarray(w, dtype=np.float32).ravel()
            layer_mean_vals.append(float(np.mean(arr)))
            layer_std_vals.append(float(np.std(arr)))
            layer_sparsity_vals.append(float(np.sum(np.abs(arr) < 1e-7) / arr.size))

        if layer_mean_vals:
            avg_mean = sum(layer_mean_vals) / len(layer_mean_vals)
            avg_std = sum(layer_std_vals) / len(layer_std_vals)
            avg_sparsity = sum(layer_sparsity_vals) / len(layer_sparsity_vals)
            all_means.append(avg_mean)
            all_stds.append(avg_std)
            all_sparsities.append(avg_sparsity)
            total_wp += pcount
            layer_summaries.append({
                "layer_name": layer.name,
                "op_type": layer.op_type,
                "param_count": pcount,
                "mean": round(avg_mean, 6),
                "std": round(avg_std, 6),
                "sparsity": round(avg_sparsity, 4),
            })

    return {
        "layers": layer_summaries,
        "global_stats": {
            "total_weight_params": total_wp,
            "overall_mean": round(float(np.mean(all_means)) if all_means else 0, 6),
            "overall_std": round(float(np.mean(all_stds)) if all_stds else 0, 6),
            "overall_sparsity": round(float(np.mean(all_sparsities)) if all_sparsities else 0, 4),
        },
    }
