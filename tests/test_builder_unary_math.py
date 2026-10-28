"""Registered elementary math shares checked Rust semantics across lowering paths."""
import math

import pytest
from groggy import Graph
from groggy.builder import AlgorithmBuilder


@pytest.mark.parametrize("name,value,kwargs,expected", [
    ("abs", -9007199254740993, {}, 9007199254740993),
    ("sqrt", 9, {}, 3.0), ("exp", 0, {}, 1.0),
    ("log", math.e, {}, 1.0), ("log", 16, {"base": 4}, 2.0),
])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("use_ir", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_unary_math_has_matching_checked_results_across_builder_modes(name, value, kwargs, expected, optimize, use_ir, atomic):
    graph = Graph()
    graph.add_nodes(3)
    builder = AlgorithmBuilder("unary math", use_ir=use_ir)
    values = builder.init_nodes(default=value)
    result = getattr(builder.core, name)(values, **kwargs)
    builder.attach_as("result", result)
    graph.induced_subgraph([0, 2]).apply(builder.build(optimize=optimize, atomic=atomic), persist=True)
    for node in [0, 2]:
        actual = graph.get_node_attr(node, "result")
        if isinstance(expected, int):
            assert isinstance(actual, int) and actual == expected
        else:
            assert actual == pytest.approx(expected)
    assert graph.get_node_attr(1, "result") is None


@pytest.mark.parametrize("name,value", [("abs", -(2**63)), ("sqrt", -1), ("exp", 1000), ("log", 0), ("log", -1), ("abs", True)])
def test_invalid_math_discards_staged_prefix_and_keeps_history(name, value):
    graph = Graph()
    graph.add_node(existing=7)
    head = graph.current_state_id
    builder = AlgorithmBuilder("invalid unary")
    builder.attach_as("prefix", builder.init_nodes(default=9))
    result = getattr(builder.core, name)(builder.init_nodes(default=value))
    builder.attach_as("result", result)
    with pytest.raises(RuntimeError):
        graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "prefix") is None
    assert graph.get_node_attr(0, "existing") == 7


@pytest.mark.parametrize("base", [0, 1, -1, True, float("inf"), float("nan")])
def test_invalid_log_base_rejects_before_empty_or_nonempty_execution(base):
    graph = Graph()
    builder = AlgorithmBuilder("invalid log base")
    result = builder.core.log(builder.init_nodes(default=2), base=base)
    builder.attach_as("result", result)
    with pytest.raises((RuntimeError, ValueError)):
        graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.current_state_id == 0


@pytest.mark.parametrize("name,value,expected", [("abs", -3, 3), ("sqrt", 9, 3), ("exp", 0, 1), ("log", 1, 0)])
def test_unary_scalar_identity_survives_fusion_and_explicit_broadcast(name, value, expected):
    graph = Graph()
    graph.add_node()
    for optimize in [False, True]:
        builder = AlgorithmBuilder("scalar math")
        reference = builder.init_nodes(default=2)
        scalar = getattr(builder.core, name)(builder.core.constant(value))
        builder.attach_as("result", reference * scalar + 1)
        graph.view().apply(builder.build(optimize=optimize), persist=True)
        assert graph.get_node_attr(0, "result") == pytest.approx(2 * expected + 1)


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("use_ir", [False, True])
def test_elementwise_minmax_preserve_exact_comparison_and_map_identity(atomic, optimize, use_ir):
    graph = Graph()
    graph.add_node()
    builder = AlgorithmBuilder("extrema", use_ir=use_ir)
    large = builder.init_nodes(default=9007199254740993)
    builder.attach_as("maximum", builder.core.max(large, 9007199254740992.0))
    builder.attach_as("minimum", builder.core.min(large, 9007199254740992.0))
    small = builder.init_nodes(default=3)
    builder.attach_as("composed", builder.core.min(small, 2) * 2 + 1)
    graph.view().apply(builder.build(optimize=optimize, atomic=atomic), persist=True)
    assert graph.get_node_attr(0, "maximum") == 9007199254740993
    assert isinstance(graph.get_node_attr(0, "maximum"), int)
    assert graph.get_node_attr(0, "minimum") == 9007199254740992.0
    assert isinstance(graph.get_node_attr(0, "minimum"), float)
    assert graph.get_node_attr(0, "composed") == 5


@pytest.mark.parametrize("name", ["min", "max"])
@pytest.mark.parametrize("optimize", [False, True])
def test_scalar_extrema_preserve_left_ties_and_fuse_with_maps(name, optimize):
    graph = Graph()
    graph.add_node()
    builder = AlgorithmBuilder("scalar extrema")
    scalar = getattr(builder.core, name)(builder.core.constant(7), 7.0)
    reference = builder.init_nodes(default=2)
    result = reference * scalar + 1
    builder.attach_as("tie", builder.core.broadcast_scalar(scalar, reference))
    builder.attach_as("result", result)
    graph.view().apply(builder.build(optimize=optimize), persist=True)
    assert graph.get_node_attr(0, "result") == 15
    assert isinstance(graph.get_node_attr(0, "tie"), int)


@pytest.mark.parametrize("value", [1e-20, -1e-20, float.fromhex("0x0.0000000000001p-1022"),
                                    -float.fromhex("0x0.0000000000001p-1022"), 1e-300])
@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_tiny_nonzero_divisors_work_across_builder_modes(value, atomic, optimize):
    graph = Graph()
    graph.add_node()
    builder = AlgorithmBuilder("tiny division")
    values = builder.init_nodes(default=value)
    builder.attach_as("ratio", values / value)
    graph.view().apply(builder.build(atomic=atomic, optimize=optimize), persist=True)
    assert graph.get_node_attr(0, "ratio") == 1.0


@pytest.mark.parametrize("zero", [0.0, -0.0])
@pytest.mark.parametrize("optimize", [False, True])
def test_true_zero_division_discards_an_atomic_prefix(zero, optimize):
    graph = Graph()
    graph.add_node()
    head = graph.current_state_id
    builder = AlgorithmBuilder("zero division")
    builder.attach_as("prefix", builder.init_nodes(default=9))
    builder.attach_as("ratio", builder.init_nodes(default=1.0) / zero)
    with pytest.raises(RuntimeError, match="division by zero"):
        graph.view().apply(builder.build(atomic=True, optimize=optimize), persist=True)
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "prefix") is None
