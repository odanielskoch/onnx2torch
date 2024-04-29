__all__ = [
    "OnnxIf"
]

from typing import Any
import types
import inspect
import textwrap

import torch
from torch import nn 
from torch.fx.graph_module import GraphModule

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

from onnx2torch.utils.common import OnnxMapping


class OnnxIf(nn.Module, OnnxToTorchModule):
    def __init__(self, then_branch: GraphModule, else_branch: GraphModule,
                 then_args: tuple[str, ...], else_args: tuple[str, ...]):
        super().__init__()
        self.then_branch = then_branch
        self.else_branch = else_branch
        # init args to arbitary value (maintains order, should be fine)
        self.then_args = [f"then_{i}" for i in range(len(then_args))]
        self.else_args = [f"else_{i}" for i in range(len(else_args))]
        
        # generate forward function 
        forward_func = (
            f"def forward(self, cond: torch.Tensor, {', '.join(self.then_args + self.else_args)}):\n" + \
            f"  assert torch.numel(cond) == 1\n" + \
            f"  if cond.item():\n" + \
            f"      return self.then_branch({', '.join(self.then_args)})\n" + \
            f"  else:\n" + \
            f"      return self.else_branch({', '.join(self.else_args)})\n"
        )
        exec(forward_func, globals(), locals())
        self.forward = types.MethodType(locals()['forward'], self)

def _onnx_if_converter(node: OnnxNode, graph: OnnxGraph, version: int) -> OperationConverterResult:
    from onnx2torch.converter import convert_graph
    #TODO: need to access input_values of graph
    then_graph: OnnxGraph = OnnxGraph(node.attributes["then_branch"].g, parent_graph=graph)
    else_graph = OnnxGraph(node.attributes["else_branch"].g, parent_graph=graph)
    then_branch: GraphModule = convert_graph(then_graph, version=version)
    else_branch: GraphModule = convert_graph(else_graph, version=version)

    then_args = tuple(then_graph.input_values)
    else_args = tuple(else_graph.input_values)
    return OperationConverterResult(
        torch_module=OnnxIf(
            then_branch=then_branch, 
            else_branch=else_branch,
            then_args=then_args,
            else_args=else_args
        ),
        onnx_mapping=OnnxMapping(
            inputs=node.input_values + then_args + else_args,
            outputs=node.output_values
        )
    )

@add_converter(operation_type='If', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    return _onnx_if_converter(node=node, graph=graph, version=11)

@add_converter(operation_type='If', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    return _onnx_if_converter(node=node, graph=graph, version=13)
