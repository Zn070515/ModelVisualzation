import onnx
import numpy as np
from onnx import helper, TensorProto
from backend.parser.onnx_parser import parse_onnx
import tempfile
import os


def build_test_onnx():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 30, 30])
    W = helper.make_tensor(
        "W", TensorProto.FLOAT, [4, 3, 3, 3],
        np.random.randn(4, 3, 3, 3).astype(np.float32),
    )
    node_conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["conv_out"], kernel_shape=[3, 3])
    node_relu = helper.make_node("Relu", inputs=["conv_out"], outputs=["Y"])
    graph = helper.make_graph([node_conv, node_relu], "test", [X], [Y], [W])
    model = helper.make_model(graph, producer_name="test-producer", opset_imports=[helper.make_opsetid("", 17)])
    return model


def test_parse_onnx_basic():
    model = build_test_onnx()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        tmp = f.name
    try:
        ir = parse_onnx(tmp)
        assert ir.format == "onnx"
        assert ir.producer == "test-producer"
        assert ir.opset_version == 17
        assert len(ir.layers) == 2
        assert ir.layers[0].op_type == "Conv"
        assert ir.layers[1].op_type == "Relu"
        assert len(ir.inputs) == 1
        assert ir.inputs[0].shape == [1, 3, 32, 32]
        assert len(ir.outputs) == 1
        assert ir.outputs[0].shape == [1, 4, 30, 30]
        assert "kernel_shape" in ir.layers[0].params
        assert ir.layers[0].params["kernel_shape"] == [3, 3]
    finally:
        os.unlink(tmp)
