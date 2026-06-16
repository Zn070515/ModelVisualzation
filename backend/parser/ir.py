from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str


@dataclass
class IRLayer:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    params: dict
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]
    weights: dict = field(default_factory=dict)

    def param_count(self) -> int:
        total = 0
        for w in self.weights.values():
            total += int(np.prod(w.shape)) if hasattr(w, 'shape') else 0
        return total


@dataclass
class IRModel:
    format: str
    producer: str
    opset_version: Optional[int]
    layers: list[IRLayer]
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]

    def total_params(self) -> int:
        return sum(layer.param_count() for layer in self.layers)
