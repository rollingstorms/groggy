"""Default optimization must preserve operands, graph effects and read ordering."""
import pytest
from groggy import Graph, Pipeline
from groggy.builder import AlgorithmBuilder
from groggy.builder.ir.graph import IRGraph
from groggy.builder.ir.nodes import AttrIRNode, CoreIRNode, ExecutionBlockNode
from groggy.builder.ir.optimizer import IROptimizer


@pytest.mark.parametrize("operation,expected", [
    (lambda a, b: (a - b) + (b - a), 0),
    (lambda a, b: (a / b) + (b / a), 5.2),
])
def test_default_optimization_preserves_noncommutative_operands(operation, expected):
    for optimize in [False, True]:
        graph = Graph()
        graph.add_node()
        builder = AlgorithmBuilder("operand positions")
        a, b = builder.init_nodes(default=10), builder.init_nodes(default=2)
        builder.attach_as("result", operation(a, b))
        graph.view().apply(builder.build(optimize=optimize), persist=True)
        assert graph.get_node_attr(0, "result") == pytest.approx(expected)


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("persist", [False, True])
def test_default_build_preserves_interleaved_property_writes_and_reads(atomic, persist):
    graph = Graph()
    graph.add_node(score=7)
    before = graph.current_state_id
    builder = AlgorithmBuilder("ordered writes")
    builder.attach_as("score", builder.init_nodes(default=2))
    builder.attach_as("copy", builder.load_attr("score"))
    builder.attach_as("score", builder.init_nodes(default=3))
    _, profile = graph.view().apply(builder.build(atomic=atomic), persist=persist, return_profile=True)
    if persist:
        assert graph.get_node_attr(0, "score") == 3
        assert graph.get_node_attr(0, "copy") == 2
        assert graph.current_state_id == before + (1 if atomic else 3)
    else:
        assert graph.current_state_id == before
        assert graph.get_node_attr(0, "score") == 7
        assert profile["outputs"]["node_attributes.copy"] == [(0, 7)]


def test_sampler_build_preserves_ordered_prefix_before_emission():
    builder = AlgorithmBuilder("ordered sampler")
    builder.attach_as("score", builder.init_nodes(default=2))
    builder.attach_as("copy", builder.load_attr("score"))
    # Check the common compile path without requiring a particular sample mode.
    sampler = builder.build_sampler(validate=False)
    assert sampler.steps == builder.steps


def test_cse_keeps_branch_positions_and_does_not_assume_property_reads_are_pure():
    graph = IRGraph()
    for node in [
        CoreIRNode("w1", "where", ["mask", "left", "right"], "one"),
        CoreIRNode("w2", "where", ["mask", "right", "left"], "two"),
        AttrIRNode("read1", "load", [], "old", attr_name="score"),
        ExecutionBlockNode("block", "message_pass", "target"),
        AttrIRNode("read2", "load", [], "new", attr_name="score"),
        CoreIRNode("result", "sub", ["new", "old"], "difference"),
    ]:
        graph.add_node(node)
    assert not IROptimizer(graph).common_subexpression_elimination()
    assert graph.get_node("result").inputs == ["new", "old"]


def test_cse_still_reuses_certified_identical_arithmetic_in_one_segment():
    graph = IRGraph()
    for node in [CoreIRNode("a", "sub", ["x", "y"], "first"),
                 CoreIRNode("b", "sub", ["x", "y"], "second"),
                 CoreIRNode("c", "add", ["first", "second"], "result")]:
        graph.add_node(node)
    assert IROptimizer(graph).common_subexpression_elimination()
    assert graph.get_node("c").inputs == ["first", "first"]


@pytest.mark.parametrize("operation", ["pow", "div", "sub", "compare", "where"])
def test_cse_signature_preserves_operand_order_for_every_ordered_operator(operation):
    graph = IRGraph()
    inputs = ["mask", "left", "right"] if operation == "where" else ["left", "right"]
    first = CoreIRNode("first", operation, inputs, "a")
    second = CoreIRNode("second", operation, list(reversed(inputs)), "b")
    assert IROptimizer(graph)._operation_signature(first) != IROptimizer(graph)._operation_signature(second)


def test_outer_atomic_run_preserves_default_compiled_child_write_order():
    graph = Graph()
    graph.add_node(score=7)
    before = graph.current_state_id
    builder = AlgorithmBuilder("ordered default child")
    builder.attach_as("score", builder.init_nodes(default=2))
    builder.attach_as("copy", builder.load_attr("score"))
    Pipeline([builder.build()])(graph.view(), atomic=True)
    assert graph.get_node_attr(0, "copy") == 2
    assert graph.current_state_id == before + 1


def test_dead_code_elimination_retains_execution_blocks_as_observable_effects():
    graph = IRGraph()
    graph.add_node(ExecutionBlockNode("block", "message_pass", "target"))
    assert not IROptimizer(graph).dead_code_elimination()
    assert [node.id for node in graph.nodes] == ["block"]
