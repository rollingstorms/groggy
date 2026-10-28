"""Power uses one checked native kernel across builder operators and lowering modes."""
import pytest
from groggy import Graph
from groggy.builder import AlgorithmBuilder


@pytest.mark.parametrize("use_ir", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_power_operator_and_explicit_method_keep_operand_order_and_selected_scope(use_ir, optimize, atomic):
    graph = Graph()
    graph.add_nodes(3)
    selected = graph.induced_subgraph([0, 2])
    builder = AlgorithmBuilder("power", use_ir=use_ir)
    a, b = builder.init_nodes(default=10), builder.init_nodes(default=2)
    builder.attach_as("result", (a ** b) + builder.core.pow(b, a))
    builder.attach_as("reverse", 2 ** b)
    selected.apply(builder.build(optimize=optimize, atomic=atomic), persist=True)
    assert graph.get_node_attr(0, "result") == 1124
    assert graph.get_node_attr(2, "reverse") == 4
    assert graph.get_node_attr(1, "result") is None


@pytest.mark.parametrize("a,b,expected", [
    (3, 34, 16677181699666569), (-2, 3, -8), (0, 0, 1),
    (2, -3, 0.125), (9.0, 0.5, 3.0),
])
@pytest.mark.parametrize("optimize", [False, True])
def test_power_preserves_checked_integer_results_and_real_float_domain(a, b, expected, optimize):
    graph = Graph()
    graph.add_node()
    builder = AlgorithmBuilder("exact power")
    values = builder.init_nodes(default=a)
    builder.attach_as("power", values ** b)
    graph.view().apply(builder.build(optimize=optimize, atomic=True), persist=True)
    assert graph.get_node_attr(0, "power") == expected
    if isinstance(expected, int):
        assert isinstance(graph.get_node_attr(0, "power"), int)


@pytest.mark.parametrize("a,b", [(2, 63), (0, -1), (-2, 0.5), (float("nan"), 2), (True, 2)])
def test_invalid_power_discards_atomic_prefix_and_preserves_history_and_ids(a, b):
    graph = Graph()
    graph.add_node(existing=7)
    head = graph.current_state_id
    builder = AlgorithmBuilder("invalid power")
    values = builder.init_nodes(default=a)
    builder.attach_as("prefix", builder.init_nodes(default=9))
    builder.attach_as("power", values ** b)
    with pytest.raises(RuntimeError):
        graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "prefix") is None
    assert graph.get_node_attr(0, "existing") == 7
    assert graph.add_node() == 1


@pytest.mark.parametrize("optimize", [False, True])
def test_scalar_power_can_be_explicitly_broadcast_and_composed(optimize):
    graph = Graph()
    graph.add_node()
    builder = AlgorithmBuilder("scalar power")
    reference = builder.init_nodes(default=0)
    scalar = builder.core.pow(3, 4)
    # Scalar powers must retain scalar identity when arithmetic fusion broadcasts.
    builder.attach_as("result", reference * scalar + 2)
    graph.view().apply(builder.build(optimize=optimize), persist=True)
    assert graph.get_node_attr(0, "result") == 2
    explicit = AlgorithmBuilder("broadcast scalar")
    reference = explicit.init_nodes(default=0)
    explicit.attach_as("power", explicit.core.broadcast_scalar(explicit.core.pow(3, 4), reference))
    graph.view().apply(explicit.build(optimize=optimize, atomic=True), persist=True)
    assert graph.get_node_attr(0, "power") == 81
