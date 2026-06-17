import numpy as np
from .ir import IRLayer, IRModel, TensorSpec


def parse_tflite(file_path: str) -> IRModel:
    from tensorflow.lite.python import schema_py_generated as schema_fb  # type: ignore[import-untyped]

    with open(file_path, "rb") as f:
        buf = bytearray(f.read())
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    return _parse_tflite_from_flatbuffer(model)


def _parse_tflite_from_flatbuffer(model) -> IRModel:
    from tensorflow.lite.python import schema_py_generated as schema_fb  # type: ignore[import-untyped]

    subgraph = model.Subgraphs(0)

    # Build tensor name map from the subgraph
    tensor_names = {}
    tensor_shapes = {}
    tensor_dtypes = {}
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        name = t.Name().decode("utf-8") if t.Name() else f"tensor_{i}"
        shape = [t.Shape(k) for k in range(t.ShapeLength())] if t.ShapeLength() > 0 else []
        dtype = _tflite_dtype(t.Type())
        tensor_names[i] = name
        tensor_shapes[i] = shape
        tensor_dtypes[i] = dtype

    # Build buffer lookup: tensor_index → numpy array
    buffer_weights: dict[int, np.ndarray] = {}
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        buf_idx = t.Buffer()
        if buf_idx > 0 and buf_idx < model.BuffersLength():
            buf = model.Buffers(buf_idx)
            raw = buf.DataAsNumpy()
            if raw is not None and hasattr(raw, "__len__") and len(raw) > 0:
                np_dtype = _tflite_to_np_dtype(t.Type())
                shape = [t.Shape(k) for k in range(t.ShapeLength())]
                if np_dtype and shape and all(s > 0 for s in shape):
                    try:
                        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                        buffer_weights[i] = arr
                    except (ValueError, TypeError):
                        pass

    layers = []
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        builtin_code = op_code.BuiltinCode()
        op_type = _tflite_builtin_op_name(builtin_code)

        inputs = []
        input_shapes = []
        for j in range(op.InputsLength()):
            idx = op.Inputs(j)
            name = tensor_names.get(idx, f"tensor_{idx}")
            inputs.append(name)
            input_shapes.append(tensor_shapes.get(idx, []))

        outputs = []
        output_shapes = []
        for j in range(op.OutputsLength()):
            idx = op.Outputs(j)
            name = tensor_names.get(idx, f"tensor_{idx}")
            outputs.append(name)
            output_shapes.append(tensor_shapes.get(idx, []))

        # Extract weights from input tensors that have buffer data
        weights = {}
        for j in range(op.InputsLength()):
            idx = op.Inputs(j)
            if idx in buffer_weights:
                tname = tensor_names.get(idx, f"tensor_{idx}")
                key = tname.rsplit("/", 1)[-1].rsplit(".", 1)[-1] if "." in tname or "/" in tname else tname
                if key in ("weight", "bias", "running_mean", "running_var"):
                    weights[key] = buffer_weights[idx]
                elif key in ("kernel", "kernel:0"):
                    weights["weight"] = buffer_weights[idx]
                elif "weight" in key.lower():
                    weights["weight"] = buffer_weights[idx]
                elif key in ("bias", "bias:0"):
                    weights["bias"] = buffer_weights[idx]
                elif "bias" in key.lower():
                    weights["bias"] = buffer_weights[idx]
                else:
                    weights[key] = buffer_weights[idx]

        layers.append(IRLayer(
            name=f"{op_type}_{i}",
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            params={},
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            weights=weights,
        ))

    # Build input/output specs
    input_indices = [subgraph.Inputs(j) for j in range(subgraph.InputsLength())]
    output_indices = [subgraph.Outputs(j) for j in range(subgraph.OutputsLength())]

    input_specs = []
    for idx in input_indices:
        name = tensor_names.get(idx, f"tensor_{idx}")
        shape = tensor_shapes.get(idx, [])
        dtype = tensor_dtypes.get(idx, "float32")
        input_specs.append(TensorSpec(name=name, shape=shape, dtype=dtype))

    output_specs = []
    for idx in output_indices:
        name = tensor_names.get(idx, f"tensor_{idx}")
        shape = tensor_shapes.get(idx, [])
        dtype = tensor_dtypes.get(idx, "float32")
        output_specs.append(TensorSpec(name=name, shape=shape, dtype=dtype))

    return IRModel(
        format="tflite",
        producer="TensorFlow Lite",
        opset_version=None,
        layers=layers,
        inputs=input_specs,
        outputs=output_specs,
    )


def _tflite_dtype(dtype: int) -> str:
    mapping = {
        0: "float32", 1: "float16", 2: "int32", 3: "uint8",
        4: "int64", 5: "string", 6: "bool", 7: "int16",
        8: "complex64", 9: "int8",
    }
    return mapping.get(dtype, f"unknown({dtype})")


def _tflite_to_np_dtype(dtype: int):
    """Map TFLite tensor type to numpy dtype."""
    mapping = {
        0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8,
        4: np.int64, 6: np.bool_, 7: np.int16, 9: np.int8,
    }
    return mapping.get(dtype)


def _tflite_builtin_op_name(code: int) -> str:
    names = {
        0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D",
        4: "DEPTHWISE_CONV_2D", 5: "DEPTH_TO_SPACE", 6: "DEQUANTIZE",
        7: "EMBEDDING_LOOKUP", 8: "FLOOR", 9: "FULLY_CONNECTED",
        10: "HASHTABLE_LOOKUP", 11: "L2_NORMALIZATION", 12: "L2_POOL_2D",
        13: "LOCAL_RESPONSE_NORMALIZATION", 14: "LOGISTIC", 15: "LSH_PROJECTION",
        16: "LSTM", 17: "MAX_POOL_2D", 18: "MUL", 19: "RELU",
        20: "RELU_N1_TO_1", 21: "RELU6", 22: "RESHAPE", 23: "RESIZE_BILINEAR",
        24: "RNN", 25: "SOFTMAX", 26: "SPACE_TO_DEPTH", 27: "SVDF",
        28: "TANH", 29: "CONCAT_EMBEDDINGS", 30: "SKIP_GRAM", 31: "CALL",
        32: "CUSTOM", 33: "EMBEDDING_LOOKUP_SPARSE", 34: "PAD",
        35: "UNIDIRECTIONAL_SEQUENCE_RNN", 36: "GATHER", 37: "BATCH_TO_SPACE_ND",
        38: "SPACE_TO_BATCH_ND", 39: "TRANSPOSE", 40: "MEAN", 41: "SUB",
        42: "DIV", 43: "SQUEEZE", 44: "UNIDIRECTIONAL_SEQUENCE_LSTM",
        45: "STRIDED_SLICE", 46: "BIDIRECTIONAL_SEQUENCE_RNN", 47: "EXP",
        48: "TOPK_V2", 49: "SPLIT", 50: "LOG_SOFTMAX",
        51: "DELEGATE", 52: "BIDIRECTIONAL_SEQUENCE_LSTM", 53: "CAST",
        54: "PRELU", 55: "MAXIMUM", 56: "ARG_MAX", 57: "MINIMUM",
        58: "LESS", 59: "NEG", 60: "PADV2", 61: "GREATER",
        62: "GREATER_EQUAL", 63: "LESS_EQUAL", 64: "SELECT", 65: "SLICE",
        66: "SIN", 67: "TRANSPOSE_CONV", 68: "SPARSE_TO_DENSE",
        69: "TILE", 70: "EXPAND_DIMS", 71: "EQUAL", 72: "NOT_EQUAL",
        73: "LOG", 74: "SUM", 75: "SQRT", 76: "RSQRT", 77: "SHAPE",
        78: "POW", 79: "ARG_MIN", 80: "FAKE_QUANT", 81: "REDUCE_PROD",
        82: "REDUCE_MAX", 83: "PACK", 84: "LOGICAL_OR", 85: "ONE_HOT",
        86: "LOGICAL_AND", 87: "LOGICAL_NOT", 88: "UNPACK", 89: "REDUCE_MIN",
        90: "FLOOR_DIV", 91: "REDUCE_ANY", 92: "SQUARE", 93: "ZEROS_LIKE",
        94: "FILL", 95: "FLOOR_MOD", 96: "RANGE", 97: "RESIZE_NEAREST_NEIGHBOR",
        98: "LEAKY_RELU", 99: "SQUARED_DIFFERENCE", 100: "MIRROR_PAD",
        101: "ABS", 102: "SPLIT_V", 103: "UNIQUE", 104: "CEIL",
        105: "REVERSE_V2", 106: "ADD_N", 107: "GATHER_ND", 108: "COS",
        109: "WHERE", 110: "RANK", 111: "ELU", 112: "REVERSE_SEQUENCE",
        113: "MATRIX_DIAG", 114: "QUANTIZE", 115: "MATRIX_SET_DIAG",
        116: "ROUND", 117: "HARD_SWISH", 118: "IF",
        119: "WHILE", 120: "NON_MAX_SUPPRESSION_V4", 121: "NON_MAX_SUPPRESSION_V5",
        122: "SCATTER_ND", 123: "SELECT_V2", 124: "DENSIFY", 125: "SEGMENT_SUM",
        126: "BATCH_MATMUL", 127: "PLACEHOLDER_FOR_GREATER_OP_CODES",
        128: "CUMSUM", 129: "CALL_ONCE", 130: "BROADCAST_TO",
        131: "RFFT2D", 132: "CONV_3D", 133: "IMAG", 134: "REAL",
        135: "COMPLEX_ABS", 136: "HASHTABLE", 137: "HASHTABLE_FIND",
        138: "HASHTABLE_IMPORT", 139: "HASHTABLE_SIZE", 140: "REDUCE_ALL",
        141: "CONV_3D_TRANSPOSE", 142: "VAR_HANDLE", 143: "READ_VARIABLE",
        144: "ASSIGN_VARIABLE", 145: "BROADCAST_ARGS", 146: "RANDOM_STANDARD_NORMAL",
        147: "BUCKETIZE", 148: "RANDOM_UNIFORM", 149: "MULTINOMIAL",
        150: "GELU", 151: "DYNAMIC_UPDATE_SLICE", 152: "RELU_0_TO_1",
        153: "UNSORTED_SEGMENT_PROD", 154: "UNSORTED_SEGMENT_MAX",
        155: "UNSORTED_SEGMENT_SUM", 156: "ATAN2", 157: "UNSORTED_SEGMENT_MIN",
        158: "SIGN",
    }
    return names.get(code, f"Op_{code}")
