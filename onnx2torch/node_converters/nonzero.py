__all__ = [
    'OnnxNonZero',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxNonZero(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def forward(self, input_tensor: torch.Tensor):  # pylint: disable=missing-function-docstring
        return torch.nonzero(input_tensor).transpose(1, 0) # shape: (1, N)


@add_converter(operation_type='NonZero', version=1)
@add_converter(operation_type='NonZero', version=9)
@add_converter(operation_type='NonZero', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxNonZero(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
