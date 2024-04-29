__all__ = [
    "OnnxSplitToSequence"
]

from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

class OnnxSplitToSequence(nn.Module, OnnxToTorchModule):
    def __init__(self, axis: int = 0, keepdims: int = 1):
        super().__init__()
        self.axis = axis 
        self.keepdims = keepdims

    def forward(self, input: torch.Tensor, split: Optional[torch.Tensor]=None) -> torch.Tensor:
        if split is None:
            splits = input.split(split_size=1, dim=self.axis)
        elif len(split.shape) == 0:
            splits = input.split(split_size=split, dim=self.axis)
        else:
            splits = input.split_with_sizes(split_sizes=tuple(split.tolist()), dim=self.axis)
        if not self.keepdims:
            for s in split:
                s.squeeze_(axis=self.axis)
        return splits

@add_converter(operation_type='SplitToSequence', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    keepdims = node.attributes.get('keepdims', 1)
    return OperationConverterResult(
        torch_module=OnnxSplitToSequence(axis=axis, keepdims=keepdims),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
