import numpy as np
from backend.parser.ir import TensorSpec, IRLayer, IRModel


def test_ir_layer_param_count():
    layer = IRLayer(
        name="conv1",
        op_type="Conv",
        inputs=["input"],
        outputs=["output"],
        params={"kernel_size": 3, "stride": 1},
        input_shapes=[[1, 3, 224, 224]],
        output_shapes=[[1, 64, 224, 224]],
        weights={"weight": np.ones((64, 3, 3, 3)), "bias": np.ones((64,))},
    )
    assert layer.param_count() == 64 * 3 * 3 * 3 + 64


def test_ir_model_total_params():
    layer1 = IRLayer(
        name="conv1", op_type="Conv",
        inputs=["in"], outputs=["out1"],
        params={}, input_shapes=[[1, 3, 8, 8]], output_shapes=[[1, 4, 6, 6]],
        weights={"weight": np.ones((4, 3, 3, 3))},
    )
    layer2 = IRLayer(
        name="conv2", op_type="Conv",
        inputs=["out1"], outputs=["out2"],
        params={}, input_shapes=[[1, 4, 6, 6]], output_shapes=[[1, 8, 4, 4]],
        weights={"weight": np.ones((8, 4, 3, 3))},
    )
    model = IRModel(
        format="onnx", producer="test", opset_version=17,
        layers=[layer1, layer2],
        inputs=[TensorSpec("in", [1, 3, 8, 8], "float32")],
        outputs=[TensorSpec("out2", [1, 8, 4, 4], "float32")],
    )
    expected = (4 * 3 * 3 * 3) + (8 * 4 * 3 * 3)
    assert model.total_params() == expected
