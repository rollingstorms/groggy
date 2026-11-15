"""Checked flow maps compose with pending topology and atomic failure recovery."""
import pytest
from groggy import Graph, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def program(*steps):
    return step_pipeline("checked flow maps", list(steps), atomic=True)


def fixture():
    graph = Graph(directed=True)
    graph.add_nodes(3)
    graph.add_edges([(0, 1), (1, 2)])
    return graph


def topology():
    return [step("graph.add_nodes", count=1, output="new"),
            step("graph.add_edges", edges=[[2, {"added": 0}]], output="new_edge"),
            step("graph.remove_edges", ids=[0])]


def attach(source, attr):
    return step("core.attach_edge_attr", source=source, attr=attr)


@pytest.mark.parametrize("outer,nested", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("scalar", [False, True])
def test_flow_updates_and_residuals_include_only_explicit_pending_map_rows(outer, nested, scalar):
    graph = fixture()
    old = graph.read_view()
    head = graph.current_state_id
    operations = [step("core.init_edges", target="flow", value=3),
                  step("core.init_scalar" if scalar else "core.init_edges", target="delta", value=2),
                  step("core.flow_update", flow="flow", delta="delta", target="updated"),
                  attach("updated", "flow"),
                  step("core.load_edge_attr", attr="flow", target="reload"),
                  step("core.init_edges", target="capacity", value=9),
                  step("core.residual_capacity", capacity="capacity", flow="reload", target="residual"),
                  attach("residual", "residual")]
    if nested:
        operations = [step("iter.loop", iterations=2, body=operations)]
    if outer:
        Pipeline([program(*topology()), program(*operations)])(graph.view(), atomic=True)
    else:
        graph.apply(program(*(topology() + operations)))
    assert sorted(graph.edge_ids) == [1, 2]
    assert [graph.get_edge_attr(e, "flow") for e in [1, 2]] == [5, 5]
    assert [graph.get_edge_attr(e, "residual") for e in [1, 2]] == [4, 4]
    assert old.edge_attr(0, "flow") is None
    assert graph.current_state_id == head + 1


@pytest.mark.parametrize("kernel", ["core.flow_update", "core.residual_capacity"])
@pytest.mark.parametrize("left,right", [(2**63 - 1, 1), (-2**63, -1)])
def test_integer_overflow_rolls_back_topology_and_prior_assignments(kernel, left, right):
    if kernel == "core.residual_capacity":
        right = -right
    graph = fixture()
    head = graph.current_state_id
    operands = {"flow": "left", "delta": "right"} if kernel == "core.flow_update" else {"capacity": "left", "flow": "right"}
    with pytest.raises(RuntimeError, match="overflow"):
        graph.apply(program(*(topology() + [
            step("core.init_edges", target="left", value=left), attach("left", "prefix"),
            step("core.init_edges", target="right", value=right),
            step(kernel, target="result", **operands), attach("result", "answer")
        ])))
    assert graph.current_state_id == head
    assert sorted(graph.edge_ids) == [0, 1]
    assert graph.get_edge_attr(1, "prefix") is None
    assert graph.add_node() == 3


@pytest.mark.parametrize("kernel", ["core.flow_update", "core.residual_capacity"])
@pytest.mark.parametrize("bad", [True, "bad", None])
def test_invalid_flow_domains_fail_without_publishing_prefix(kernel, bad):
    graph = fixture()
    head = graph.current_state_id
    operands = {"flow": "bad", "delta": "good"} if kernel == "core.flow_update" else {"capacity": "good", "flow": "bad"}
    with pytest.raises(RuntimeError, match="numeric"):
        graph.apply(program(*(topology() + [
            step("core.init_edges", target="good", value=1), attach("good", "prefix"),
            step("core.init_edges", target="bad", value=bad),
            step(kernel, target="result", **operands)
        ])))
    assert graph.current_state_id == head
    assert graph.get_edge_attr(1, "prefix") is None


@pytest.mark.parametrize("kernel,operands,expected", [
    ("core.flow_update", {"flow": "large", "delta": "two"}, 2**53 + 3),
    ("core.residual_capacity", {"capacity": "large", "flow": "two"}, 2**53 - 1),
])
def test_integer_results_remain_exact_beyond_float_precision(kernel, operands, expected):
    graph = fixture()
    graph.apply(program(*(topology() + [
        step("core.init_edges", target="large", value=2**53 + 1),
        step("core.init_scalar" if kernel == "core.flow_update" else "core.init_edges", target="two", value=2),
        step(kernel, target="result", **operands), attach("result", "answer")
    ])))
    for edge in [1, 2]:
        assert graph.get_edge_attr(edge, "answer") == expected
        assert isinstance(graph.get_edge_attr(edge, "answer"), int)


def test_explicit_old_flow_map_is_not_silently_rebound_to_pending_edge_ids():
    graph = fixture()
    head = graph.current_state_id
    with pytest.raises(RuntimeError):
        graph.apply(program(*([step("core.init_edges", target="old", value=1)] + topology() + [
            step("core.init_scalar", target="delta", value=1),
            step("core.flow_update", flow="old", delta="delta", target="result"),
            attach("result", "answer")
        ])))
    assert graph.current_state_id == head
    assert graph.get_edge_attr(1, "answer") is None


@pytest.mark.parametrize("kernel", ["core.flow_update", "core.residual_capacity"])
def test_finite_operands_with_nonfinite_result_fail_before_residual_filtering(kernel):
    graph = fixture()
    head = graph.current_state_id
    operands = {"flow": "left", "delta": "right"} if kernel == "core.flow_update" else {"capacity": "left", "flow": "right"}
    with pytest.raises(RuntimeError, match="finite"):
        graph.apply(program(*(topology() + [
            step("core.init_edges", target="left", value=1e308 if kernel == "core.flow_update" else -1e308),
            step("core.init_edges", target="right", value=1e308),
            step(kernel, target="result", **operands), attach("result", "answer")
        ])))
    assert graph.current_state_id == head
    assert graph.get_edge_attr(1, "answer") is None


@pytest.mark.parametrize("kernel", ["core.flow_update", "core.residual_capacity"])
def test_positive_subnormal_flow_results_are_not_dropped_by_epsilon_rules(kernel):
    graph = fixture()
    tiny = float.fromhex("0x0.0000000000001p-1022")
    operands = {"flow": "tiny", "delta": "zero"} if kernel == "core.flow_update" else {"capacity": "tiny", "flow": "zero"}
    graph.apply(program(*(topology() + [
        step("core.init_edges", target="tiny", value=tiny),
        step("core.init_edges", target="zero", value=-0.0),
        step(kernel, target="result", **operands), attach("result", "answer")
    ])))
    assert [graph.get_edge_attr(e, "answer") for e in [1, 2]] == [tiny, tiny]
