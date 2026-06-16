import io

import numpy as np


def collect_activations(model, sample_bytes: bytes, model_id: str = "",
                        layer_names: list[str] | None = None,
                        model_path: str | None = None) -> dict:
    sample = _load_sample(sample_bytes)
    selected = set(layer_names or [])
    activations = []

    ort_outputs = None
    if model_path and model_path.endswith(".onnx"):
        ort_outputs = _run_onnx_with_intermediate(model_path, sample)

    current = sample
    for idx, layer in enumerate(model.layers):
        if selected and layer.name not in selected:
            continue
        if ort_outputs is not None and layer.name in ort_outputs:
            current = ort_outputs[layer.name]
            method = "onnxruntime"
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

        # Collect all intermediate node output names and add them as graph outputs
        existing_outputs = {o.name for o in graph.output}
        for node in graph.node:
            for output_name in node.output:
                if output_name and output_name not in existing_outputs:
                    value_info = onnx.helper.make_tensor_value_info(
                        output_name, onnx.TensorProto.FLOAT, []
                    )
                    graph.output.append(value_info)
                    existing_outputs.add(output_name)

        # Save temp model with all intermediate outputs
        import tempfile
        import os
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
        # Only use actual weight/bias tensors for scale, skip running stats
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
    return int(arr.shape[1]) if arr.ndim >= 2 else int(arr.shape[0])


def _dead_indices(arr: np.ndarray) -> list[int]:
    if arr.ndim < 2:
        return [int(i) for i, value in enumerate(arr.ravel()) if abs(float(value)) < 1e-7]
    axes = tuple(index for index in range(arr.ndim) if index != 1)
    channel_mean = np.mean(np.abs(arr), axis=axes)
    return [int(index) for index, value in enumerate(channel_mean) if float(value) < 1e-7]
