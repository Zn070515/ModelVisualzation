import numpy as np


def simulate_quantization(model, model_id: str = "", bits: int = 8) -> dict:
    if bits not in (8, 16):
        raise ValueError("Only 8-bit integer and 16-bit float simulation are supported")

    layers = []
    all_errors = []
    for layer in model.layers:
        layer_weights = {}
        for name, weight in layer.weights.items():
            if not hasattr(weight, "shape"):
                continue
            arr = np.asarray(weight, dtype=np.float32)
            stats = _simulate_tensor(arr, bits)
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
        "layers": layers,
        "summary": {
            "overall_mean_abs_err": round(mean_abs, 8),
            "overall_rmse": round(rmse, 8),
            "worst_layer": worst[0],
            "worst_layer_rmse": round(float(worst[1]), 8),
            "sensitive_layers": sorted(set(sensitive)),
        },
    }


def _simulate_tensor(arr: np.ndarray, bits: int) -> dict:
    if bits == 16:
        restored = arr.astype(np.float16).astype(np.float32)
        return _tensor_result(arr, restored, -65504, 65504, 1.0, 0)

    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    scale = max_abs / qmax if max_abs > 0 else 1.0
    quantized = np.clip(np.round(arr / scale), qmin, qmax)
    restored = (quantized * scale).astype(np.float32)
    return _tensor_result(arr, restored, qmin, qmax, scale, 0)


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
