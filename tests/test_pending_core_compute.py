"""Pending compute kernels keep selection, typed inputs and one acceptance."""
import math

import pytest
from groggy import Graph, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def program(*steps):
    return step_pipeline("pending core computation", list(steps), atomic=True)


def fixture(directed=False):
    graph = Graph(directed=directed)
    graph.add_nodes(7)
    graph.remove_nodes([1, 3])
    graph.add_edges([(0, 2), (2, 4), (2, 4), (4, 4), (5, 6)])
    for edge in graph.edge_ids:
        graph.set_edge_attr(edge, "ordinal", edge)
    return graph


def topology():
    return [step("graph.add_nodes", count=1, output="new"),
            step("graph.add_edges", edges=[[0, {"added": 0}], [{"added": 0}, 4]], output="edges", attributes={"ordinal": 100}),
            step("graph.set_edge_endpoints", revisions=[[0, 4, 0]]),
            step("graph.remove_nodes", ids=[2])]


def attach(source, attr="answer", edges=False):
    return step("core.attach_edge_attr" if edges else "core.attach_node_attr", source=source, attr=attr)


def run(graph, operations, outer=False, nested=False):
    selected = graph.induced_subgraph([0, 2, 4])
    if nested:
        operations = [step("iter.loop", iterations=1, body=operations)]
    if outer:
        return Pipeline([program(*topology()), program(*operations)])(selected, atomic=True)
    return selected.apply(program(*(topology() + operations)))


@pytest.mark.parametrize("outer,nested", [(False, False), (True, False), (True, True)])
def test_filters_order_and_topk_read_final_membership_and_staged_values(outer, nested):
    graph = fixture()
    old = graph.read_view()
    head = graph.current_state_id
    run(graph, [step("core.load_node_attr", attr="node_id", target="ids"),
                attach("ids", "score"),
                step("core.filter_nodes_by_attr", attr="score", predicate={"type": "gt", "value": {"type": "Int", "value": 3}}, target="mask"),
                attach("mask", "selected"),
                step("core.sort_nodes_by_attr", attr="score", order="asc", target="ranks"),
                attach("ranks", "rank"),
                step("core.top_k", source="ids", k=2, order="desc", target="top"),
                attach("top", "top"),
                step("core.load_edge_attr", attr="ordinal", target="edge_ids"),
                attach("edge_ids", "score", edges=True),
                step("core.filter_edges_by_attr", attr="score", predicate={"type": "gt", "value": {"type": "Int", "value": 3}}, target="edges"),
                attach("edges", "selected", edges=True)], outer, nested)
    assert [graph.get_node_attr(n, "selected") for n in [0, 4, 7]] == [None, 1, 1]
    assert [graph.get_node_attr(n, "rank") for n in [0, 4, 7]] == [0, 1, 2]
    assert [graph.get_node_attr(n, "top") for n in [0, 4, 7]] == [None, 1, 0]
    assert [graph.get_edge_attr(e, "selected") for e in [0, 3, 5, 6]] == [None, None, 1, 1]
    assert graph.get_node_attr(5, "score") is None
    assert old.edge_endpoints(0) == (0, 2)
    assert old.node_attr(4, "score") is None
    assert graph.current_state_id == head + 1


@pytest.mark.parametrize("kernel,params,expected", [
    ("core.normalize_values", {"method": "sum"}, [0, 4 / 11, 7 / 11]),
    ("core.normalize_node_values", {"method": "max"}, [0, 4 / 7, 1]),
    ("core.normalize_values", {"method": "minmax"}, [0, 4 / 7, 1]),
    ("core.standardize", {}, [-11 / math.sqrt(74), 1 / math.sqrt(74), 10 / math.sqrt(74)]),
    ("core.clip", {"min": 2.0, "max": 5.0}, [2, 4, 5]),
])
@pytest.mark.parametrize("nested", [False, True])
def test_numerical_steps_consume_explicit_pending_maps(kernel, params, expected, nested):
    graph = fixture()
    run(graph, [step("core.load_node_attr", attr="node_id", target="ids"),
                step(kernel, source="ids", target="result", **params), attach("result")], nested=nested)
    assert [graph.get_node_attr(n, "answer") for n in [0, 4, 7]] == pytest.approx(expected)


def test_edge_normalization_uses_edge_map_without_reinterpreting_as_nodes():
    graph = fixture()
    run(graph, [step("core.load_edge_attr", attr="ordinal", target="ids"),
                step("core.normalize_values", source="ids", target="result", method="sum"),
                attach("result", edges=True)])
    assert [graph.get_edge_attr(e, "answer") for e in [0, 3, 5, 6]] == pytest.approx([0, 3 / 203, 100 / 203, 100 / 203])


@pytest.mark.parametrize("direction,expected", [("in", [8, 22, 0]), ("out", [14, 8, 8]), ("undirected", [22, 22, 8])])
@pytest.mark.parametrize("directed", [False, True])
def test_fused_neighbors_follow_pending_endpoints_and_explicit_direction(direction, expected, directed):
    graph = fixture(directed)
    run(graph, [step("core.load_node_attr", attr="node_id", target="ids"),
                step("core.init_nodes", target="two", value=2),
                step("graph.fused_neighbor_mul_agg", values="ids", scalars="two", direction=direction, target="result"),
                attach("result")], outer=True, nested=True)
    assert [graph.get_node_attr(n, "answer") for n in [0, 4, 7]] == expected


@pytest.mark.parametrize("kernel,params", [
    ("core.fused_axpy", {"a": "two", "x": "ids", "b": "one", "y": "two"}),
    ("core.fused_madd", {"a": "ids", "b": "two", "c": "two"}),
])
def test_fused_arithmetic_includes_new_rows_without_outside_owner_rows(kernel, params):
    graph = fixture()
    run(graph, [step("core.load_node_attr", attr="node_id", target="ids"),
                step("core.init_scalar", target="one", value=1),
                step("core.init_nodes", target="two", value=2),
                step(kernel, target="result", **params), attach("result")])
    assert [graph.get_node_attr(n, "answer") for n in [0, 4, 7]] == [2, 10, 16]
    assert graph.get_node_attr(5, "answer") is None


@pytest.mark.parametrize("kind,population", [("nodes", [0, 4, 7]), ("edges", [0, 3, 5, 6])])
@pytest.mark.parametrize("count", [0, 2])
def test_seeded_masks_and_reservoir_share_the_same_pending_population(kind, population, count):
    graph = fixture()
    edges = kind == "edges"
    run(graph, [step("core.sample_" + kind, count=count, seed=43, target="sample"),
                attach("sample", "sample", edges),
                step("core.sample_" + kind, fraction=1.0, seed=7, target="all"),
                step("core.reservoir_sample", source="all", entity_type=kind, k=count, seed=43, target="again"),
                attach("again", "again", edges)], outer=True, nested=True)
    read = graph.get_edge_attr if edges else graph.get_node_attr
    sampled = [entity for entity in population if read(entity, "sample") == 1]
    assert len(sampled) == count
    assert sampled == [entity for entity in population if read(entity, "again") == 1]
    if not edges:
        assert read(5, "sample") is None


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("strategy,params", [("singleton", {}), ("degree", {}), ("random", {"k": 3, "seed": 43})])
def test_community_primitives_match_equivalent_accepted_graph(directed, strategy, params):
    pending = fixture(directed)
    reference = Graph(directed=directed)
    reference.add_nodes(8)
    reference.remove_nodes([1, 2, 3, 5, 6])
    reference.add_edges([(4, 0), (4, 4), (0, 7), (7, 4)])
    operations = [step("core.community_seed", strategy=strategy, target="labels", **params),
                  attach("labels", "seed"),
                  step("core.label_propagate_step", labels="labels", target="propagated"),
                  attach("propagated", "propagated"),
                  step("core.modularity_gain", partition="labels", target="gain"),
                  attach("gain", "gain")]
    run(pending, operations, outer=True, nested=True)
    reference.apply(program(*operations))
    for attr in ["seed", "propagated", "gain"]:
        assert [pending.get_node_attr(n, attr) for n in [0, 4, 7]] == pytest.approx([reference.get_node_attr(n, attr) for n in [0, 4, 7]])


@pytest.mark.parametrize("kernel,params", [
    ("core.top_k", {"k": 0}),
    ("core.reservoir_sample", {"k": 0, "entity_type": "nodes"}),
])
def test_removed_source_rows_fail_even_for_zero_results_and_roll_back(kernel, params):
    graph = fixture()
    head = graph.current_state_id
    with pytest.raises(RuntimeError, match="outside active selection"):
        graph.induced_subgraph([0, 2, 4]).apply(program(*(
            [step("core.load_node_attr", attr="node_id", target="old")] + topology() +
            [step(kernel, source="old", target="result", **params)])))
    assert graph.current_state_id == head
    assert graph.edge_endpoints(0) == (0, 2)
    assert graph.add_node() == 7


def test_late_numeric_failure_discards_topology_and_prior_staged_properties():
    graph = fixture()
    head = graph.current_state_id
    with pytest.raises(RuntimeError, match="numeric"):
        run(graph, [step("core.init_nodes", value=9, target="prefix"), attach("prefix", "prefix"),
                    step("core.init_nodes", value="bad", target="invalid"),
                    step("core.normalize_values", source="invalid", target="result")], nested=True)
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "prefix") is None
    assert sorted(graph.node_ids) == [0, 2, 4, 5, 6]
    assert graph.add_node() == 7


@pytest.mark.parametrize("layout", ["nodes", "column", "edges"])
@pytest.mark.parametrize("outer", [False, True])
def test_temporal_decay_keeps_pending_entity_kind_and_in_place_loop_inputs(layout, outer):
    graph = fixture(True)
    head = graph.current_state_id
    if layout == "column":
        load = step("core.community_seed", strategy="singleton", target="values")
    elif layout == "edges":
        load = step("core.init_edges", value=8, target="values")
    else:
        load = step("core.load_node_attr", attr="node_id", target="values")
    operations = [load, step("iter.loop", iterations=2, body=[
        step("temporal.decay", attr="values", output="values", half_life=2.0, time_delta=2.0)]),
        attach("values", edges=layout == "edges")]
    run(graph, operations, outer=outer)
    if layout == "edges":
        assert [graph.get_edge_attr(e, "answer") for e in [0, 3, 5, 6]] == [2.0] * 4
    else:
        assert [graph.get_node_attr(n, "answer") for n in [0, 4, 7]] == [0.0, 1.0, 1.75]
        assert graph.get_node_attr(5, "answer") is None
    assert graph.current_state_id == head + 1


@pytest.mark.parametrize("edges", [False, True])
def test_temporal_decay_rejects_removed_rows_from_explicit_old_maps(edges):
    graph = fixture()
    head = graph.current_state_id
    with pytest.raises(RuntimeError, match="outside the active selection"):
        graph.induced_subgraph([0, 2, 4]).apply(program(*(
            [step("core.init_edges" if edges else "core.init_nodes", value=8, target="old")] + topology() +
            [step("temporal.decay", attr="old", output="result", half_life=2.0, time_delta=1.0)])))
    assert graph.current_state_id == head
    assert graph.add_node() == 7


@pytest.mark.parametrize("invalid", [True, None, "bad"])
def test_temporal_decay_domain_failure_discards_prior_pending_assignments(invalid):
    graph = fixture()
    head = graph.current_state_id
    with pytest.raises(RuntimeError, match="finite numeric"):
        run(graph, [step("core.init_nodes", value=8, target="prefix"), attach("prefix", "prefix"),
                    step("core.init_nodes", value=invalid, target="values"),
                    step("temporal.decay", attr="values", output="result", half_life=2.0, time_delta=1.0)])
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "prefix") is None
    assert graph.add_node() == 7
