import io
import os

import numpy as np


def collect_activations(model, sample_bytes: bytes, model_id: str = "",
                        layer_names: list[str] | None = None,
                        model_path: str | None = None) -> dict:
    sample = _load_sample(sample_bytes)
    selected = set(layer_names or [])
    activations = []

    ort_outputs = None
    pytorch_outputs = None
    tflite_outputs = None

    if model_path:
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".onnx":
            ort_outputs = _run_onnx_with_intermediate(model_path, sample)
        elif ext in (".pt", ".pth"):
            pytorch_outputs = _run_pytorch_hook_forward(model_path, sample)
        elif ext == ".tflite":
            tflite_outputs = _run_tflite_interpreter(model_path, sample)

    current = sample
    for idx, layer in enumerate(model.layers):
        if selected and layer.name not in selected:
            continue
        if ort_outputs is not None and layer.name in ort_outputs:
            current = ort_outputs[layer.name]
            method = "onnxruntime"
        elif pytorch_outputs is not None and layer.name in pytorch_outputs:
            current = pytorch_outputs[layer.name]
            method = "pytorch_hook"
        elif tflite_outputs is not None:
            match = _find_tflite_output(tflite_outputs, layer, idx)
            if match is not None:
                current = match
                method = "tflite_interpreter"
            else:
                current = _synthetic_forward(layer, current)
                method = "synthetic"
        else:
            current = _synthetic_forward(layer, current)
            method = "synthetic"
        stats = _stats(current)
        dead = _dead_indices(current)
        activations.append({
            "layer_name": layer.name,
            "output_shape": [int(dim) for dim in current.shape],
            "stats": stats,
            "dead_neurons_pct": round(100 * len(dead) / max(_channel_count(current), 1), 4),
            "dead_neuron_indices": dead[:100],
            "saturation_pct": round(float(np.mean(np.abs(current) > 6.0) * 100), 4),
            "method": method,
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


def _run_onnx_with_intermediate(model_path: str, sample: np.ndarray) -> dict[str, np.ndarray] | None:
    """Run ONNX model and collect per-node outputs by modifying the graph."""
    import onnx
    import onnxruntime as ort

    try:
        proto = onnx.load(model_path)
        graph = proto.graph

        existing_outputs = {o.name for o in graph.output}
        for node in graph.node:
            for output_name in node.output:
                if output_name and output_name not in existing_outputs:
                    value_info = onnx.helper.make_tensor_value_info(
                        output_name, onnx.TensorProto.FLOAT, []
                    )
                    graph.output.append(value_info)
                    existing_outputs.add(output_name)

        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        onnx.save(proto, tmp_path)

        try:
            session = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
            input_info = session.get_inputs()
            if not input_info:
                return None
            input_name = input_info[0].name
            input_shape = input_info[0].shape

            if hasattr(sample, "reshape") and len(input_shape) > 1:
                try:
                    dynamic_shape = tuple(
                        d if isinstance(d, (int,)) and d > 0 else 1
                        for d in input_shape
                    )
                    batch = sample.reshape(dynamic_shape).astype(np.float32)
                except Exception:
                    batch = sample.reshape((1, -1)).astype(np.float32)
            else:
                batch = sample.astype(np.float32)

            output_names = [o.name for o in session.get_outputs()]
            outputs = session.run(output_names, {input_name: batch})
            return {name: np.asarray(val, dtype=np.float32) for name, val in zip(output_names, outputs)}
        finally:
            os.unlink(tmp_path)
    except Exception:
        return None


def _run_pytorch_hook_forward(model_path: str, sample: np.ndarray) -> dict[str, np.ndarray] | None:
    try:
        import torch
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    if isinstance(loaded, dict):
        return None
    model = loaded if isinstance(loaded, torch.nn.Module) else None
    if model is None:
        return None

    outputs: dict[str, np.ndarray] = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if out is not None:
                t = out if isinstance(out, torch.Tensor) else out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor):
                    outputs[name] = t.detach().cpu().numpy().astype(np.float32)
        return hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        model.eval()
        with torch.no_grad():
            if sample.ndim <= 3:
                inp = torch.from_numpy(sample).float().unsqueeze(0)
            else:
                inp = torch.from_numpy(sample).float()
            model(inp)
    except Exception:
        pass
    finally:
        for h in hooks:
            h.remove()
    return outputs if outputs else None


def _run_tflite_interpreter(model_path: str, sample: np.ndarray) -> dict[str, np.ndarray] | None:
    try:
        import tensorflow as tf
    except ImportError:
        return None

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        tensor_details = interpreter.get_tensor_details()
        input_details = interpreter.get_input_details()

        if not input_details:
            return None

        inp = sample.astype(np.float32)
        inp_shape = input_details[0]["shape"]
        try:
            if len(inp.shape) < len(inp_shape):
                reshape_target = [d if d > 0 else 1 for d in inp_shape]
                inp = inp.reshape(reshape_target)
        except Exception:
            pass
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()

        outputs = {}
        for d in tensor_details:
            name = d.get("name") or f"tensor_{d['index']}"
            outputs[name] = interpreter.get_tensor(d["index"]).astype(np.float32)
        return outputs
    except Exception:
        return None


def _find_tflite_output(outputs: dict[str, np.ndarray], layer, idx: int) -> np.ndarray | None:
    """Match a TFLite tensor output to an IR layer by name or operator index."""
    # TFLite tensor names may be prefixed differently, try multiple patterns
    candidates = [
        layer.name,
        f"{layer.op_type}_{idx}",
        # Common TFLite naming patterns
        f"sequential/{layer.name}/output",
        f"sequential/{layer.name}/BiasAdd",
        f"sequential/{layer.name}/Relu",
        layer.name.replace("CONV_2D_", "sequential/conv2d_"),
    ]
    # Also try matching against actual output tensor names
    for op_output_name in layer.outputs:
        candidates.append(op_output_name)

    for key, value in outputs.items():
        key_lower = key.lower()
        for cand in candidates:
            if cand and cand.lower() in key_lower:
                return np.asarray(value, dtype=np.float32)
        # Also try: layer name parts embedded in key
        if layer.name.lower() in key_lower.replace("/", " ").replace("_", " "):
            return np.asarray(value, dtype=np.float32)

    # Fallback: match by operator index
    suffix = f"_output_{idx}"
    for key, value in outputs.items():
        if key.endswith(suffix) or f"_{idx}" in key.rsplit("/", 1)[-1]:
            return np.asarray(value, dtype=np.float32)
    return None


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
        scale_candidates = []
        for key, w in layer.weights.items():
            if not hasattr(w, "shape"):
                continue
            if key in ("running_mean", "running_var", "num_batches_tracked"):
                continue
            scale_candidates.append(np.asarray(w, dtype=np.float32))
        if scale_candidates:
            weight_scale = float(np.mean([np.std(v) for v in scale_candidates])) or 1.0
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
    if arr.ndim == 0:
        return 1
    return int(arr.shape[1]) if arr.ndim >= 2 else int(arr.shape[0])


def _dead_indices(arr: np.ndarray) -> list[int]:
    if arr.ndim < 2:
        return [int(i) for i, value in enumerate(arr.ravel()) if abs(float(value)) < 1e-7]
    axes = tuple(index for index in range(arr.ndim) if index != 1)
    channel_mean = np.mean(np.abs(arr), axis=axes)
    return [int(index) for index, value in enumerate(channel_mean) if float(value) < 1e-7]
