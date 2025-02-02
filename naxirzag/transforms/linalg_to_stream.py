import os
import importlib.resources

from xdsl.dialects import builtin, memref
from xdsl.dialects.linalg import GenericOp
from xdsl.context import MLContext
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import ShapedType, IntegerType, IntAttr
from dataclasses import dataclass

import yaml

import zigzag.inputs.hardware
import zigzag.inputs.mapping
import zigzag.api


class LinalgToStreamTranslator(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, generic_op: GenericOp, rewriter: PatternRewriter):

        if len(generic_op.outputs) != 1:
            return
        if not isinstance(generic_op.outputs[0].type, ShapedType):
            return

        # extract output op and relevant indexing maps
        output_op = generic_op.outputs[0]
        output_map = generic_op.indexing_maps.data[-1].data

        # make some assertions on correct inputs of the linalg generic
        shaped_inputs = [
            (op, map.data)
            for (op, map) in zip(generic_op.inputs, generic_op.indexing_maps.data[:-1])
            if isinstance(op.type, ShapedType)
        ]
        if len(shaped_inputs) != 2:
            return

        # input a, input b, output
        operands = [shaped_inputs[0][0], shaped_inputs[1][0], output_op]
        indexing_maps = [shaped_inputs[0][1], shaped_inputs[1][1], output_map]

        zigzag_description = dict()
        zigzag_description["id"] = 0

        # for now, set operator to default type
        zigzag_description["operator_type"] = "default"

        # construct equation
        output_access = "O"
        for i in range(len(indexing_maps[-1].results)):
            map = indexing_maps[-1].results[i]
            assert isinstance(map, AffineDimExpr)
            output_access += f"[{str(map)}]"

        input_i_access = "I"
        for i in range(len(indexing_maps[0].results)):
            map = indexing_maps[0].results[i]
            assert isinstance(map, AffineDimExpr)
            input_i_access += f"[{str(map)}]"

        input_w_access = "W"
        for i in range(len(indexing_maps[1].results)):
            map = indexing_maps[1].results[i]
            assert isinstance(map, AffineDimExpr)
            input_w_access += f"[{str(map)}]"

        # assume MAC
        zigzag_description["equation"] = (
            f"{output_access} += {input_i_access} * {input_w_access}"
        )

        # extract dimension_relations
        # for matmul, this is empty
        zigzag_description["dimension_relations"] = []

        # extract loop bounds by evaluating the inverse affine map
        # with the memref shapes as input
        results: list[AffineExpr] = []
        results.extend(indexing_maps[0].results)
        results.extend(indexing_maps[1].results)
        results.extend(indexing_maps[2].results)

        combined_affine_map = AffineMap(3, 0, tuple(results))
        inverse_map = combined_affine_map.inverse_permutation()

        memref_shapes = [shape.data for op in operands for shape in op.type.shape.data]
        iteration_bounds = inverse_map.eval(memref_shapes, [])

        zigzag_description["loop_dim_size"] = dict()

        for i, bound in enumerate(iteration_bounds):
            zigzag_description["loop_dim_size"][f"D{i}"] = bound

        # extract operand precision
        widths = []
        for op in operands:
            element_type = op.type.get_element_type()
            if isinstance(element_type, IntegerType):
                widths.append(element_type.width.data)
            else:
                widths.append(element_type.get_bitwidth)

        zigzag_description["operand_precision"] = dict()
        zigzag_description["operand_precision"]["O"] = widths[-1]
        zigzag_description["operand_precision"]["O_final"] = widths[-1]
        zigzag_description["operand_precision"]["W"] = widths[0]
        zigzag_description["operand_precision"]["I"] = widths[1]

        # operand source (use default of no source for now)
        zigzag_description["operand_source"] = dict()
        zigzag_description["operand_source"]["W"] = 0
        zigzag_description["operand_source"]["I"] = 0

        # affects last two indices of input I
        workload = [zigzag_description]

        with open("workload.yaml", "w") as f:
            f.write(yaml.dump(workload, sort_keys=False))


        hardware_path = importlib.resources.files('zigzag.inputs.hardware') / 'gemm_l1.yaml'
        mapping_path = importlib.resources.files('zigzag.inputs.mapping') / 'gemm_l1.yaml'

        # add stream id attribute to the generic op
        generic_op.attributes["zigzag_stream_id"] = IntAttr(0)

@dataclass(frozen=True)
class LinalgToStream(ModulePass):

    name = "linalg-to-stream"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:

        PatternRewriteWalker(LinalgToStreamTranslator(), apply_recursively=False).rewrite_module(op)
