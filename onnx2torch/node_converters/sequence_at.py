__all__ = [
    "OnnxSequenceAt"
]

from typing import List

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

class OnnxSequenceAt(nn.Module, OnnxToTorchModule):
    def forward(self, input_sequence: List[torch.Tensor], position: torch.Tensor) -> torch.Tensor:
        assert position.numel() == 1
        return input_sequence[position]

@add_converter(operation_type='SequenceAt', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxSequenceAt(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
