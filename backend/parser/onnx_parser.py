import onnx
from onnx import shape_inference
import numpy as np
from .ir import IRLayer, IRModel, TensorSpec


def parse_onnx(file_path: str) -> IRModel:
    model = onnx.load(file_path)
    inferred = shape_inference.infer_shapes(model)
    graph = inferred.graph

    value_info = {}
    for vi in graph.input:
        shape = [d.dim_value if d.dim_value else -1 for d in vi.type.tensor_type.shape.dim]
        value_info[vi.name] = (shape, _dtype_str(vi.type.tensor_type.elem_type))
    for vi in graph.output:
        shape = [d.dim_value if d.dim_value else -1 for d in vi.type.tensor_type.shape.dim]
        value_info[vi.name] = (shape, _dtype_str(vi.type.tensor_type.elem_type))
    for vi in graph.value_info:
        shape = [d.dim_value if d.dim_value else -1 for d in vi.type.tensor_type.shape.dim]
        value_info[vi.name] = (shape, _dtype_str(vi.type.tensor_type.elem_type))

    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = onnx.numpy_helper.to_array(init)

    layers = []
    for node in graph.node:
        params = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                params[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                params[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.STRING:
                params[attr.name] = attr.s.decode("utf-8")
            elif attr.type == onnx.AttributeProto.FLOAT:
                params[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.FLOATS:
                params[attr.name] = list(attr.floats)

        input_shapes = []
        for inp in node.input:
            if inp in value_info:
                input_shapes.append(value_info[inp][0])

        output_shapes = []
        for out in node.output:
            if out in value_info:
                output_shapes.append(value_info[out][0])

        weights = {}
        for inp in node.input:
            if inp in initializers:
                weights[inp] = initializers[inp]

        layers.append(IRLayer(
            name=node.name or node.op_type,
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            params=params,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            weights=weights,
        ))

    input_specs = [
        TensorSpec(name=inp.name, shape=(
            [d.dim_value if d.dim_value else -1
             for d in inp.type.tensor_type.shape.dim]
        ), dtype=_dtype_str(inp.type.tensor_type.elem_type))
        for inp in graph.input
    ]
    output_specs = [
        TensorSpec(name=out.name, shape=(
            [d.dim_value if d.dim_value else -1
             for d in out.type.tensor_type.shape.dim]
        ), dtype=_dtype_str(out.type.tensor_type.elem_type))
        for out in graph.output
    ]

    return IRModel(
        format="onnx",
        producer=model.producer_name or "unknown",
        opset_version=model.opset_import[0].version if model.opset_import else None,
        layers=layers,
        inputs=input_specs,
        outputs=output_specs,
    )


def _dtype_str(elem_type: int) -> str:
    mapping = {1: "float32", 6: "int32", 7: "int64", 10: "float16", 11: "float64"}
    return mapping.get(elem_type, f"unknown({elem_type})")
