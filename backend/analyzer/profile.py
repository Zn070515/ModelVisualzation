import numpy as np


def compute_profile(ir_model):
    layer_profiles = []
    total_params = 0
    total_flops = 0

    for layer in ir_model.layers:
        pcount = layer.param_count()
        flops = _estimate_flops(layer)
        total_params += pcount
        total_flops += flops
        layer_profiles.append({
            "name": layer.name,
            "op_type": layer.op_type,
            "params_count": pcount,
            "flops": flops,
        })

    memory_mb = round((total_params * 4 * 2) / (1024 * 1024), 2)

    return {
        "total_params": total_params,
        "total_flops": total_flops,
        "memory_mb": memory_mb,
        "layers": layer_profiles,
    }


def _estimate_flops(layer) -> int:
    op = layer.op_type.lower()
    if "conv" in op:
        return _conv_flops(layer)
    if "linear" in op or "fully" in op or "gemm" in op:
        return _linear_flops(layer)
    return 0


def _conv_flops(layer) -> int:
    if not layer.output_shapes:
        return 0
    out_shape = layer.output_shapes[0]
    if len(out_shape) < 2:
        return 0
    n, c_out = out_shape[0], out_shape[1]
    spatial = int(np.prod(out_shape[2:])) if len(out_shape) > 2 else 1

    weight = layer.weights.get("weight")
    if weight is None and layer.weights:
        weight = list(layer.weights.values())[0]
    if weight is not None and hasattr(weight, 'shape') and len(weight.shape) >= 2:
        c_in = int(weight.shape[1])
        k = int(np.prod(weight.shape[2:]))
    else:
        c_in = layer.input_shapes[0][1] if layer.input_shapes and len(layer.input_shapes[0]) >= 2 else 1
        k = 9

    return n * c_out * c_in * k * spatial * 2


def _linear_flops(layer) -> int:
    weight = layer.weights.get("weight")
    if weight is not None and hasattr(weight, 'shape') and len(weight.shape) == 2:
        return int(weight.shape[0]) * int(weight.shape[1]) * 2
    return 0
