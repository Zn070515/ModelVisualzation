from __future__ import annotations

from collections import Counter


def trace_conversion_chain(
    models: list,
    labels: list[str] | None = None,
    model_ids: list[str] | None = None,
) -> dict:
    if labels is None:
        labels = model_ids if model_ids else [f"model_{i}" for i in range(len(models))]
    if model_ids is None:
        model_ids = [""] * len(models)

    stages: list[dict] = []
    for model, label, mid in zip(models, labels, model_ids):
        op_counts = Counter(layer.op_type for layer in model.layers)
        stages.append({
            "label": label,
            "model_id": mid,
            "format": model.format,
            "layer_count": len(model.layers),
            "total_params": model.total_params(),
            "op_types": sorted(op_counts.keys()),
            "op_counts": dict(op_counts),
            "total_ops": sum(op_counts.values()),
        })

    transitions: list[dict] = []
    for i in range(len(models) - 1):
        a_ops = {layer.op_type for layer in models[i].layers}
        b_ops = {layer.op_type for layer in models[i + 1].layers}

        added = sorted(b_ops - a_ops)
        removed = sorted(a_ops - b_ops)
        common = a_ops & b_ops

        renamed = _detect_renames(removed, added)
        renamed_sources = {r["from"] for r in renamed}
        renamed_targets = {r["to"] for r in renamed}

        transitions.append({
            "from": labels[i],
            "to": labels[i + 1],
            "from_id": model_ids[i],
            "to_id": model_ids[i + 1],
            "added_ops": [a for a in added if a not in renamed_targets],
            "removed_ops": [r for r in removed if r not in renamed_sources],
            "renamed_ops": renamed,
            "common_ops": sorted(common),
            "layer_count_delta": len(models[i + 1].layers) - len(models[i].layers),
            "param_count_delta": models[i + 1].total_params() - models[i].total_params(),
        })

    total_op_loss = sum(len(t["added_ops"]) + len(t["removed_ops"]) for t in transitions)

    return {
        "model_ids": model_ids,
        "stages": stages,
        "transitions": transitions,
        "summary": {
            "total_steps": len(transitions),
            "total_op_loss": total_op_loss,
            "preserved_op_count": sum(len(t["common_ops"]) for t in transitions) if transitions else 0,
        },
    }


def _detect_renames(removed: list[str], added: list[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    remaining_added = list(added)
    remaining_removed = list(removed)
    for r in list(remaining_removed):
        for a in list(remaining_added):
            if _fuzzy_match(r, a):
                result.append({"from": r, "to": a})
                remaining_removed.remove(r)
                remaining_added.remove(a)
                break
    return result


def _fuzzy_match(a: str, b: str) -> bool:
    al = a.lower().replace("_", "").replace("2d", "").replace("1d", "").replace("3d", "")
    bl = b.lower().replace("_", "").replace("2d", "").replace("1d", "").replace("3d", "")
    return al == bl and a != b
