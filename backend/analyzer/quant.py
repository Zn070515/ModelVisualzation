from __future__ import annotations

import numpy as np


def simulate_quantization(
    model,
    model_id: str = "",
    bits: int = 8,
    per_channel: bool = False,
    unsigned: bool = False,
) -> dict:
    if bits not in (4, 8, 16):
        raise ValueError("Supported bits: 4, 8, 16")

    mode = "fp16" if bits == 16 else ("uint" if unsigned else "int")
    layers: list[dict] = []
    all_errors: list[tuple[str, float, float]] = []

    for layer in model.layers:
        layer_weights: dict[str, dict] = {}
        for name, weight in layer.weights.items():
            if not hasattr(weight, "shape"):
                continue
            arr = np.asarray(weight, dtype=np.float32)
            stats = _simulate_tensor(arr, bits, per_channel, unsigned, layer.op_type)
            layer_weights[name] = stats
            all_errors.append((layer.name, stats["error"]["rmse"], stats["error"]["mean_abs_err"]))
        if layer_weights:
            layers.append({"layer_name": layer.name, "op_type": layer.op_type, "weights": layer_weights})

    worst = max(all_errors, key=lambda item: item[1], default=("", 0.0, 0.0))
    mean_abs = float(np.mean([item[2] for item in all_errors])) if all_errors else 0.0
    rmse = float(np.mean([item[1] for item in all_errors])) if all_errors else 0.0
    sensitive = [name for name, layer_rmse, _ in all_errors if layer_rmse >= max(rmse * 1.5, 1e-12)]

    return {
        "model_id": model_id,
        "bits": bits,
        "mode": mode,
        "per_channel": per_channel,
        "layers": layers,
        "summary": {
            "overall_mean_abs_err": round(mean_abs, 8),
            "overall_rmse": round(rmse, 8),
            "worst_layer": worst[0],
            "worst_layer_rmse": round(float(worst[1]), 8),
            "sensitive_layers": sorted(set(sensitive)),
        },
    }


def _simulate_tensor(arr: np.ndarray, bits: int, per_channel: bool, unsigned: bool, op_type: str = "") -> dict:
    if bits == 16:
        restored = arr.astype(np.float16).astype(np.float32)
        return _tensor_result(arr, restored, -65504, 65504, 1.0, 0)

    if per_channel:
        return _per_channel_quant(arr, bits, unsigned, op_type)
    return _per_tensor_quant(arr, bits, unsigned)


def _per_tensor_quant(arr: np.ndarray, bits: int, unsigned: bool) -> dict:
    if unsigned:
        qmax = (2 ** bits) - 1
        qmin = 0
    else:
        qmax = (2 ** (bits - 1)) - 1
        qmin = -qmax

    if unsigned:
        max_val = float(np.max(arr)) if arr.size else 0.0
        min_val = float(np.min(arr)) if arr.size else 0.0
        scale = (max_val - min_val) / qmax if (max_val - min_val) > 0 else 1.0
        zero_point = int(round(-min_val / scale)) if scale else 0
        zero_point = max(qmin, min(qmax, zero_point))
    else:
        max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
        scale = max_abs / qmax if max_abs > 0 else 1.0
        zero_point = 0

    quantized = np.clip(np.round(arr / scale) + zero_point, qmin, qmax).astype(np.float32)
    restored = ((quantized - zero_point) * scale).astype(np.float32)
    return _tensor_result(arr, restored, qmin, qmax, scale, zero_point)


def _per_channel_quant(arr: np.ndarray, bits: int, unsigned: bool, op_type: str = "") -> dict:
    if unsigned:
        qmax = (2 ** bits) - 1
        qmin = 0
    else:
        qmax = (2 ** (bits - 1)) - 1
        qmin = -qmax

    if arr.ndim <= 1:
        return _per_tensor_quant(arr, bits, unsigned)

    channel_axis = 0
    op_lower = op_type.lower()
    if "convtranspose" in op_lower and arr.ndim >= 4:
        channel_axis = 1

    if channel_axis == 0 and arr.ndim == 4 and arr.shape[3] > arr.shape[0] and arr.shape[3] > arr.shape[1]:
        channel_axis = 3

    c_out = arr.shape[channel_axis]
    moved = np.moveaxis(arr, channel_axis, 0)
    reshaped = moved.reshape(c_out, -1)
    scales = np.zeros(c_out, dtype=np.float32)
    zero_points = np.zeros(c_out, dtype=np.int32)
    restored_flat = np.zeros(reshaped.shape, dtype=np.float32)

    for c in range(c_out):
        row = reshaped[c]
        if unsigned:
            max_val = float(np.max(row)) if row.size else 0.0
            min_val = float(np.min(row)) if row.size else 0.0
            scales[c] = (max_val - min_val) / qmax if (max_val - min_val) > 0 else 1.0
            zp = int(round(-min_val / scales[c])) if scales[c] else 0
            zero_points[c] = max(qmin, min(qmax, zp))
            restored_flat[c] = (np.clip(np.round(row / scales[c]) + zero_points[c], qmin, qmax) - zero_points[c]) * scales[c]
        else:
            max_abs = float(np.max(np.abs(row))) if row.size else 0.0
            scales[c] = max_abs / qmax if max_abs > 0 else 1.0
            zero_points[c] = 0
            restored_flat[c] = np.clip(np.round(row / scales[c]), qmin, qmax) * scales[c]

    restored_moved = restored_flat.reshape(moved.shape)
    restored = np.moveaxis(restored_moved, 0, channel_axis).astype(np.float32)

    diff = restored - arr
    abs_diff = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
    signal = float(np.mean(arr ** 2)) if arr.size else 0.0
    noise = float(np.mean(diff ** 2)) if diff.size else 0.0
    snr = 10 * np.log10(signal / noise) if noise > 0 and signal > 0 else 99.0
    return {
        "quant_min": qmin,
        "quant_max": qmax,
        "scale": round(float(np.mean(scales)), 10),
        "zero_point": 0,
        "per_channel_scales": [round(float(s), 10) for s in scales],
        "per_channel_zero_points": [int(z) for z in zero_points],
        "error": {
            "max_abs_err": round(float(np.max(abs_diff)) if abs_diff.size else 0.0, 8),
            "mean_abs_err": round(float(np.mean(abs_diff)) if abs_diff.size else 0.0, 8),
            "rmse": round(rmse, 8),
            "snr_db": round(float(snr), 4),
        },
        "per_channel_error": _per_channel_error(arr, restored),
    }


def _tensor_result(original: np.ndarray, restored: np.ndarray, qmin: int, qmax: int, scale: float, zero_point: int) -> dict:
    diff = restored - original
    abs_diff = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
    signal = float(np.mean(original ** 2)) if original.size else 0.0
    noise = float(np.mean(diff ** 2)) if diff.size else 0.0
    snr = 10 * np.log10(signal / noise) if noise > 0 and signal > 0 else 99.0
    return {
        "quant_min": qmin,
        "quant_max": qmax,
        "scale": round(float(scale), 10),
        "zero_point": zero_point,
        "error": {
            "max_abs_err": round(float(np.max(abs_diff)) if abs_diff.size else 0.0, 8),
            "mean_abs_err": round(float(np.mean(abs_diff)) if abs_diff.size else 0.0, 8),
            "rmse": round(rmse, 8),
            "snr_db": round(float(snr), 4),
        },
        "per_channel_error": _per_channel_error(original, restored),
    }


def _per_channel_error(original: np.ndarray, restored: np.ndarray) -> list[float]:
    if original.ndim == 0:
        return [0.0]
    if original.ndim == 1:
        errors = np.abs(restored - original)
    else:
        axes = tuple(range(1, original.ndim))
        errors = np.mean(np.abs(restored - original), axis=axes)
    return [round(float(value), 8) for value in np.asarray(errors).ravel().tolist()]
