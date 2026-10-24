"""Expression and in-place transformation parity with atomic property publication."""
import json

import pytest
from groggy import Graph
from groggy.algorithms.base import AlgorithmHandle
from groggy.builder import AlgorithmBuilder


class AtomicSteps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        from groggy import AttrValue
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({"atomic": True, "steps": self.steps}))}}


def const(value):
    return {"type": "const", "value": value}


def binary(op, left, right):
    return {"type": "binary_op", "op": op, "left": left, "right": right}


def call(func, *args):
    return {"type": "call", "func": func, "args": list(args)}


def mapping(expr, **extra):
    return {"id": "core.map_nodes", "params": {"source": "values", "target": "result", "expr": expr, **extra}}


def attach(source="result", attr="answer"):
    return {"id": "core.attach_node_attr", "params": {"source": source, "attr": attr}}


def load():
    return {"id": "core.load_node_attr", "params": {"attr": "score", "target": "values"}}


def run_expression(expr, persist=True):
    graph = Graph()
    graph.add_nodes([{}, {}])
    before = graph.current_state_id
    _, profile = graph.view().apply(AtomicSteps([mapping(expr), attach()]), persist=persist, return_profile=True)
    result = graph.get_node_attr(0, "answer") if persist else profile["outputs"]["node_attributes.answer"][0][1]
    assert graph.current_state_id == before + int(persist)
    return result


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("op,a,b,expected", [
    ("add", 7, 3, 10), ("sub", 7, 3, 4), ("mul", 7, 3, 21),
    ("div", 7, 2, 3.5), ("mod", 7, 3, 1), ("pow", 2, 10, 1024),
    ("div", 1e-20, 1e-20, 1), ("gt", 2**53 + 1, float(2**53), True),
    ("eq", 2**53 + 1, float(2**53), False), ("eq", 2, 2.0, True),
    ("add", "a", "b", "ab"),
])
def test_expression_operators_are_checked_and_publish_once(op, a, b, expected, persist):
    assert run_expression(binary(op, const(a), const(b)), persist) == expected


@pytest.mark.parametrize("expr", [
    binary("add", const(2**63 - 1), const(1)),
    binary("sub", const(-2**63), const(1)),
    binary("mul", const(2**62), const(4)),
    binary("mod", const(-2**63), const(-1)),
    binary("pow", const(2), const(63)),
    binary("div", const(1), const(0.0)),
    binary("div", const(1.0), const(0)),
    {"type": "unary_op", "op": "abs", "arg": const(-2**63)},
    {"type": "unary_op", "op": "sqrt", "arg": const(-1)},
    {"type": "unary_op", "op": "exp", "arg": const(1000)},
    call("sum", const([2**63 - 1, 1])),
])
def test_expression_errors_discard_all_prefix_attachments(expr):
    graph = Graph()
    graph.add_node()
    before = graph.current_state_id
    steps = [{"id": "core.init_nodes", "params": {"target": "prefix", "value": 9}}, attach("prefix", "prefix"), mapping(expr), attach()]
    with pytest.raises(RuntimeError):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "prefix") is None
    assert graph.get_node_attr(0, "answer") is None


@pytest.mark.parametrize("expr,expected", [
    (call("mean", const([2**63 - 1, 2**63 - 1])), float(2**63 - 1)),
    (call("mean", const([1e308, 1e308])), 1e308),
    (call("sum", const([1e16, 1.0, -1e16])), 1),
    (call("mode", const([-3.0, 2.0])), -3.0),
    (call("max", const(2**53 + 1), const(float(2**53))), 2**53 + 1),
    (binary("and", const(False), binary("div", const(1), const(0))), False),
    (binary("or", const(True), binary("div", const(1), const(0))), True),
    ({"type": "if", "condition": const(True), "then_expr": const(7), "else_expr": binary("div", const(1), const(0))}, 7),
])
def test_expression_reductions_and_lazy_branches(expr, expected):
    assert run_expression(expr) == expected


def test_expression_attribute_reads_include_prior_staged_properties():
    persist = True
    graph = Graph()
    graph.add_nodes([{"score": 1}, {"score": 2}])
    before = graph.current_state_id
    steps = [load(), attach("values", "stage"), mapping(binary("add", {"type": "attr", "name": "stage"}, const(10))), attach()]
    _, profile = graph.view().apply(AtomicSteps(steps), persist=persist, return_profile=True)
    assert [graph.get_node_attr(n, "answer") for n in range(2)] == [11, 12]
    assert graph.current_state_id == before + 1


@pytest.mark.parametrize("asynchronous,expected", [(False, [2, 4, 2]), (True, [2, 5, 5])])
def test_async_mapping_preserves_id_order_and_source_visibility(asynchronous, expected):
    graph = Graph()
    graph.add_nodes([{"score": n} for n in [1, 2, 3]])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    expr = call("sum", call("neighbor_values", {"type": "var", "name": "values"}))
    graph.view().apply(AtomicSteps([load(), mapping(expr, async_update=asynchronous), attach(), attach("values", "source")]), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == expected
    assert [graph.get_node_attr(n, "source") for n in range(3)] == (expected if asynchronous else [1, 2, 3])


@pytest.mark.parametrize("use_ir", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_public_map_parser_and_lowering_preserve_exact_integer_constants(use_ir, optimize, atomic):
    graph = Graph()
    graph.add_nodes([{"score": 2**53}, {"score": 2**53 + 1}])
    builder = AlgorithmBuilder("checked_map", use_ir=use_ir)
    source = builder.load_attr("score")
    result = builder.map_nodes("x + 1", inputs={"x": source})
    builder.attach_as("answer", result)
    graph.view().apply(builder.build(atomic=atomic, optimize=optimize), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(2)] == [2**53 + 1, 2**53 + 2]


@pytest.mark.parametrize("expr", ["x + 1 trailing", "sum(x[neighbors(node)]) + 1", "x[neighbors(node)] trailing"])
def test_parser_rejects_unconsumed_expression_text(expr):
    from groggy.expr_parser import parse_expression
    with pytest.raises(ValueError):
        parse_expression(expr)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("tie_break,expected", [("lowest", 2**53), ("highest", 2**53 + 1), ("keep", 2**53)])
def test_neighbor_label_ties_keep_exact_numeric_order_and_parallel_incidence(ordered, tie_break, expected):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": 0}, {"score": 2**53}, {"score": 2**53 + 1}])
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    step = {"id": "core.neighbor_mode_update", "params": {"target": "values", "include_self": False, "ordered": ordered, "tie_break": tie_break, "output": "result"}}
    graph.view().apply(AtomicSteps([load(), step, attach()]), persist=True)
    assert graph.get_node_attr(0, "answer") == expected


@pytest.mark.parametrize("params", [{"tie_break": "typo"}, {"tie_break": 1}, {"ordered": "yes"}, {"include_self": 1}, {"output": 3}])
def test_neighbor_controls_reject_malformed_values(params):
    graph = Graph()
    graph.add_node()
    with pytest.raises(RuntimeError):
        graph.view().apply(AtomicSteps([load(), {"id": "core.neighbor_mode_update", "params": {"target": "values", **params}}]), persist=True)


def test_transient_attachments_do_not_simulate_expression_property_reads():
    graph = Graph()
    graph.add_nodes([{"score": 1}])
    before = graph.current_state_id
    with pytest.raises(RuntimeError, match="attribute 'staged' not found"):
        graph.view().apply(AtomicSteps([load(), attach("values", "staged"), mapping({"type": "attr", "name": "staged"}), attach()]), persist=False)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "staged") is None


@pytest.mark.parametrize("func,expected", [("neighbor_count", 4), ("neighbors", [0, 1, 1, 2]), ("neighbor_values", [1, 10, 20])])
def test_expression_selected_neighbor_contracts_include_self_and_parallel_edges(func, expected):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": n} for n in [1, 10, 20]])
    for target in [0, 1, 1, 2]:
        graph.add_edge(0, target)
    expr = call(func, {"type": "var", "name": "values"}) if func == "neighbor_values" else call(func)
    graph.view().apply(AtomicSteps([load(), mapping(expr), attach()]), persist=True)
    assert graph.get_node_attr(0, "answer") == expected


@pytest.mark.parametrize("use_ir", [False, True])
def test_public_async_and_inplace_operations_compose_with_atomic_publication(use_ir):
    graph = Graph()
    graph.add_nodes([{"score": 1}, {"score": 2}, {"score": 3}])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    before = graph.current_state_id
    builder = AlgorithmBuilder("async_map", use_ir=use_ir)
    source = builder.load_attr("score")
    result = builder.map_nodes("sum(x[neighbors(node)])", inputs={"x": source}, async_update=True)
    target = builder.init_nodes(default=0)
    updated = builder.core.update_in_place(result, target=target, ordered=True)
    builder.attach_as("answer", updated)
    graph.view().apply(builder.build(atomic=True), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == [2, 5, 5]
    assert graph.current_state_id == before + 1


@pytest.mark.parametrize("ordered", [False, True])
def test_parallel_edges_vote_separately_in_neighbor_mode(ordered):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": n} for n in [0, 7, 2]])
    for target in [1, 1, 2]:
        graph.add_edge(0, target)
    step = {"id": "core.neighbor_mode_update", "params": {"target": "values", "output": "result", "include_self": False, "ordered": ordered}}
    graph.view().apply(AtomicSteps([load(), step, attach()]), persist=True)
    assert graph.get_node_attr(0, "answer") == 7


@pytest.mark.parametrize("tie_break", ["lowest", "highest"])
def test_numerically_equal_typed_label_ties_are_deterministic(tie_break):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": 0}, {"score": 2}, {"score": 2.0}])
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    step = {"id": "core.neighbor_mode_update", "params": {"target": "values", "output": "result", "include_self": False, "tie_break": tie_break}}
    for _ in range(4):
        graph.view().apply(AtomicSteps([load(), step, attach()]), persist=True)
        assert isinstance(graph.get_node_attr(0, "answer"), float)


@pytest.mark.parametrize("expression,expected", [("count(x[neighbors(node)])", 2), ("neighbors(node)", [1, 2])])
def test_documented_public_neighbor_expressions_execute(expression, expected):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": n} for n in [1, 2, 3]])
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    builder = AlgorithmBuilder("neighbor_expression")
    source = builder.load_attr("score")
    result = builder.map_nodes(expression, inputs={"x": source})
    builder.attach_as("answer", result)
    graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.get_node_attr(0, "answer") == expected
