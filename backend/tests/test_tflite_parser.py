import numpy as np
import tensorflow as tf
from backend.parser.tflite_parser import parse_tflite
import tempfile
import os


def build_test_tflite():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(4, 3, use_bias=False),
        tf.keras.layers.ReLU(),
    ])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def test_parse_tflite_basic():
    tflite_bytes = build_test_tflite()
    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
        f.write(tflite_bytes)
        tmp = f.name
    try:
        ir = parse_tflite(tmp)
        assert ir.format == "tflite"
        assert len(ir.layers) >= 1
        op_types = [l.op_type for l in ir.layers]
        assert any("CONV" in t.upper() for t in op_types)
        assert len(ir.inputs) == 1
        assert len(ir.outputs) == 1
    finally:
        os.unlink(tmp)
