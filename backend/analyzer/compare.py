from collections import Counter

import numpy as np


def compare_models(model_a, model_b, model_a_id: str = "", model_b_id: str = "") -> dict:
    a_types = Counter(layer.op_type for layer in model_a.layers)
    b_types = Counter(layer.op_type for layer in model_b.layers)

    # Build name→layer maps
    a_by_name = {layer.name: layer for layer in model_a.layers}
    b_by_name = {layer.name: layer for layer in model_b.layers}

    all_names = sorted(set(a_by_name) | set(b_by_name))
    a_only = sorted(set(a_by_name) - set(b_by_name))
    b_only = sorted(set(b_by_name) - set(a_by_name))
    common_names = sorted(set(a_by_name) & set(b_by_name))

    layer_diffs = []
    for name in all_names:
        la = a_by_name.get(name)
        lb = b_by_name.get(name)
        if la and lb:
            same_type = la.op_type == lb.op_type
            same_params = la.param_count() == lb.param_count()
            status = "same" if same_type and same_params else "changed"
        elif la:
            status = "removed"
        else:
            status = "added"

        layer_diffs.append({
            "name": name,
            "status": status,
            "a": _layer_brief(la),
            "b": _layer_brief(lb),
            "param_delta": (lb.param_count() if lb else 0) - (la.param_count() if la else 0),
        })

    # Add weight diffs for common layers
    weight_diffs = []
    for name in common_names:
        la = a_by_name[name]
        lb = b_by_name[name]
        for wname in set(la.weights.keys()) | set(lb.weights.keys()):
            wa = la.weights.get(wname)
            wb = lb.weights.get(wname)
            if wa is None or wb is None:
                continue
            if not hasattr(wa, 'shape') or not hasattr(wb, 'shape'):
                continue
            arr_a = np.asarray(wa, dtype=np.float32).ravel()
            arr_b = np.asarray(wb, dtype=np.float32).ravel()
            if arr_a.size == 0 or arr_b.size == 0:
                continue
            weight_diffs.append({
                "layer_name": name,
                "weight_name": wname,
                "mean_diff": round(float(np.mean(arr_b) - np.mean(arr_a)), 6),
                "std_diff": round(float(np.std(arr_b) - np.std(arr_a)), 6),
            })

    param_a = model_a.total_params()
    param_b = model_b.total_params()
    changed = sum(1 for item in layer_diffs if item["status"] != "same")

    all_types = sorted(set(a_types) | set(b_types))
    op_type_diff = [
        {"op_type": op_type, "a": a_types[op_type], "b": b_types[op_type], "delta": b_types[op_type] - a_types[op_type]}
        for op_type in all_types
    ]

    max_len = max(len(model_a.layers), len(model_b.layers))
    similarity = 1 - changed / max(max_len, 1)

    return {
        "model_a_id": model_a_id,
        "model_b_id": model_b_id,
        "summary": {
            "layer_count_a": len(model_a.layers),
            "layer_count_b": len(model_b.layers),
            "layer_count_delta": len(model_b.layers) - len(model_a.layers),
            "param_count_a": param_a,
            "param_count_b": param_b,
            "param_count_delta": param_b - param_a,
            "changed_layers": changed,
            "a_only_layers": a_only,
            "b_only_layers": b_only,
            "common_layer_count": len(common_names),
            "similarity": round(similarity, 4),
        },
        "op_type_diff": op_type_diff,
        "layers": layer_diffs,
        "weight_diffs": weight_diffs,
    }


def _layer_brief(layer):
    if layer is None:
        return None
    return {
        "name": layer.name,
        "op_type": layer.op_type,
        "params": layer.param_count(),
        "input_shapes": layer.input_shapes,
        "output_shapes": layer.output_shapes,
    }
