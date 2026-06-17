from __future__ import annotations

import numpy as np


def extract_attention(model, model_id: str = "") -> dict:
    """Extract attention layer weights from IRModel for visualization."""
    attention_layers = []
    for layer in model.layers:
        op_lower = layer.op_type.lower()
        if "attention" not in op_lower and "multihead" not in op_lower:
            # Also check by name for layers the parser labeled differently but
            # that have attention-like weights (q/k/v projections)
            weight_keys = [k.lower() for k in layer.weights]
            has_qkv = any("q_proj" in k or "k_proj" in k or "v_proj" in k
                         or "in_proj" in k for k in weight_keys)
            if not has_qkv:
                continue

        layer_data = _build_attention_layer(layer)
        if layer_data:
            attention_layers.append(layer_data)

    return {
        "model_id": model_id,
        "layers": attention_layers,
    }


def _build_attention_layer(layer) -> dict | None:
    weights = layer.weights
    head_data = _infer_heads(weights)
    if head_data is None:
        return None

    num_heads, head_dim, embed_dim = head_data
    projection_weights = {}
    for key, arr in weights.items():
        arr_f32 = np.asarray(arr, dtype=np.float32)
        if arr_f32.ndim < 2:
            continue
        rows, cols = arr_f32.shape[0], arr_f32.shape[1]
        projection_weights[key] = {
            "shape": list(arr_f32.shape),
            # Slice into per-head blocks for the first 8 heads (or fewer)
            "head_slices": _slice_heads(arr_f32, num_heads, head_dim, embed_dim, key),
        }

    return {
        "layer_name": layer.name,
        "op_type": layer.op_type,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "embed_dim": embed_dim,
        "projections": projection_weights,
    }


def _infer_heads(weights: dict) -> tuple[int, int, int] | None:
    """Infer (num_heads, head_dim, embed_dim) from weight shapes."""
    # Try to find a projection weight
    for key in ("in_proj_weight", "q_proj_weight", "k_proj_weight",
                 "v_proj_weight", "out_proj.weight", "weight"):
        w = weights.get(key)
        if w is not None and hasattr(w, "shape") and w.ndim >= 2:
            dim = w.shape[-1]
            # Common patterns: embed_dim = 512, 768, 1024, 4096
            # num_heads = 8, 12, 16, 32
            for num_heads in (32, 16, 12, 8, 4, 2):
                if dim % num_heads == 0:
                    head_dim = dim // num_heads
                    if 16 <= head_dim <= 256:
                        return num_heads, head_dim, dim
            # Fallback: assume 8 heads
            return 8, dim // 8, dim

    # If no projection weight found, try any 2D weight
    for w in weights.values():
        if hasattr(w, "shape") and w.ndim >= 2:
            dim = w.shape[-1]
            return 8, dim // 8, dim

    return None


def _slice_heads(arr: np.ndarray, num_heads: int, head_dim: int,
                 embed_dim: int, key: str) -> list[dict]:
    """Slice weight matrix into per-head sub-matrices."""
    slices = []
    max_heads = min(num_heads, 8)  # Limit to first 8 heads for response size

    rows, cols = arr.shape

    # Determine slicing orientation based on key
    if "in_proj" in key:
        # Combined QKV: shape (3*embed_dim, embed_dim)
        # Split into Q, K, V then split by heads
        for h in range(max_heads):
            start = h * head_dim
            end = start + head_dim
            if end > min(rows, cols):
                break
            block = arr[start:end, start:end]
            slices.append({
                "head_index": h,
                "shape": list(block.shape),
                "values": _flatten_block(block, 16),
            })
    elif "q_proj" in key or "k_proj" in key or "v_proj" in key or "out_proj" in key:
        # Per-head projection: shape (embed_dim, embed_dim) or (embed_dim, head_dim)
        # Take the top-left corner or slice by heads
        for h in range(max_heads):
            start = h * head_dim
            end = start + head_dim
            if end > min(rows, cols):
                break
            block = arr[start:end, :head_dim]
            slices.append({
                "head_index": h,
                "shape": list(block.shape),
                "values": _flatten_block(block, 16),
            })
    else:
        # Generic weight: slice by heads along first dimension
        for h in range(max_heads):
            start = h * head_dim
            end = start + head_dim
            if end > rows:
                break
            block = arr[start:end, :head_dim] if head_dim <= cols else arr[start:end, :cols]
            slices.append({
                "head_index": h,
                "shape": list(block.shape),
                "values": _flatten_block(block, 16),
            })

    return slices


def _flatten_block(block: np.ndarray, max_dim: int) -> list[list[float]]:
    """Downsample a block to at most max_dim x max_dim for response size."""
    if block.shape[0] > max_dim or block.shape[1] > max_dim:
        r_step = max(1, block.shape[0] // max_dim)
        c_step = max(1, block.shape[1] // max_dim)
        block = block[::r_step, ::c_step]
    return block.astype(float).tolist()
