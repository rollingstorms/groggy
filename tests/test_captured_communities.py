"""Community reads use frozen values and membership, without graph publication."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


NUMERIC = ("louvain", "leiden", "infomap", "girvan_newman")
ALL = NUMERIC + ("label_propagation",)


def fixture(directed=False):
    graph = gr.Graph(directed=directed)
    graph.add_nodes(8)
    for source, target in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4),
                           (4, 5), (5, 3), (0, 1), (1, 1), (5, 7)]:
        edge = graph.add_edge(source, target)
        graph.set_edge_attr(edge, "length", 1.0)
    for node in range(8):
        graph.set_node_attr(node, "seed", "left" if node < 3 else "right")
    return graph


def run(view, method):
    if method == "leiden":
        return view.leiden(seed=19)
    if method == "infomap":
        return view.infomap(seed=19, num_trials=3, max_iter=30)
    if method == "girvan_newman":
        return view.girvan_newman(num_levels=4, weight_attr="length")
    if method == "label_propagation":
        return view.label_propagation(seed_attr="seed", seed=19)
    return view.louvain()


@pytest.mark.parametrize("method", ALL)
def test_community_results_are_owned_selected_rows_and_do_not_publish(method):
    graph = fixture()
    view = graph.read_view().select_nodes(list(range(7)))
    before = graph.current_state_id
    rows = run(view, method)
    assert [node for node, _ in rows] == list(range(7))  # includes the isolate
    assert graph.current_state_id == before
    assert all(not view.has_node_attribute(node, "community") for node in view.node_ids)
    if method == "label_propagation":
        assert all(label in ("left", "right") for _, label in rows)
    else:
        assert all(isinstance(label, int) for _, label in rows)
    rows.clear()
    assert len(run(view, method)) == 7


@pytest.mark.parametrize("method", ALL)
def test_community_reads_outlive_mutations_and_owner_and_run_on_workers(method):
    graph = fixture()
    # Exact edge selection omits an external endpoint and one parallel occurrence.
    view = graph.read_view().select_edges([0, 1, 2, 3, 4, 5, 6, 8])
    expected = run(view, method)
    graph.set_node_attr(0, "seed", None)
    graph.set_edge_attr(0, "length", -1)
    graph.remove_node(2)
    del graph
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: run(view, method), range(4)))
    assert results == [expected] * 4


@pytest.mark.parametrize("method", ALL)
@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_one_node_community_reads(method, count):
    graph = gr.Graph()
    graph.add_nodes(count)
    assert len(run(graph.read_view(), method)) == count


@pytest.mark.parametrize("method", ("louvain", "leiden", "girvan_newman"))
def test_undirected_community_algorithms_reject_directed_captures(method):
    view = fixture(directed=True).read_view()
    with pytest.raises((ValueError, RuntimeError), match="undirected"):
        run(view, method)


@pytest.mark.parametrize("method", ("infomap", "label_propagation"))
def test_direction_aware_communities_accept_directed_captures(method):
    view = fixture(directed=True).read_view()
    assert len(run(view, method)) == view.node_count()


@pytest.mark.parametrize("value", [True, -1, 1.5, "2", 2**100])
@pytest.mark.parametrize("method,field", [
    ("louvain", "max_iter"), ("leiden", "max_phases"),
    ("infomap", "num_trials"), ("infomap", "flow_max_iter"),
    ("girvan_newman", "num_levels"), ("label_propagation", "max_iter"),
])
def test_counts_reject_nonintegral_negative_boolean_and_overflow(method, field, value):
    with pytest.raises((ValueError, TypeError, OverflowError, RuntimeError)):
        getattr(fixture().read_view(), method)(**{field: value})


@pytest.mark.parametrize("value", [True, -1, 1.5, "2", 2**100])
@pytest.mark.parametrize("method", ("leiden", "infomap", "label_propagation"))
def test_optional_seeds_are_unsigned_integers(method, value):
    with pytest.raises((ValueError, TypeError, OverflowError, RuntimeError)):
        getattr(fixture().read_view(), method)(seed=value)


@pytest.mark.parametrize("method,options", [
    ("louvain", {"max_iter": 0}), ("louvain", {"resolution": float("inf")}),
    ("leiden", {"theta": 0}), ("infomap", {"flow_max_iter": 0}),
    ("infomap", {"teleportation": float("nan")}),
    ("girvan_newman", {"modularity_threshold": -1}),
    ("label_propagation", {"tolerance": float("nan")}),
])
def test_invalid_domains_fail_without_mutating_owner(method, options):
    graph = fixture()
    before = graph.current_state_id
    with pytest.raises((ValueError, RuntimeError)):
        getattr(graph.read_view(), method)(**options)
    assert graph.current_state_id == before


def test_frozen_weight_validation_only_considers_selected_instances():
    graph = fixture()
    graph.set_edge_attr(7, "length", -1)
    view = graph.read_view()
    with pytest.raises((ValueError, RuntimeError)):
        view.girvan_newman(num_levels=0, weight_attr="length")
    selected = view.select_edges([0, 1, 2])
    assert len(selected.girvan_newman(num_levels=0, weight_attr="length")) == 3


def test_temporal_seed_values_and_weights_survive_corrections_and_durable_reopen(tmp_path):
    path = tmp_path / "communities.ggj"
    with gr.DurableGraph.create(path) as graph:
        batch = gr.GraphBatch(effective_at=10)
        a = batch.add_node(seed="old")
        b = batch.add_node(seed="old")
        batch.add_edge(a, b, length=1.0)
        graph.apply_batch(batch, graph.current_state_id, "initial", "test")
        captured = graph.read_view_at(15)
        expected = {method: run(captured, method) for method in ALL}
        batch = gr.GraphBatch(effective_at=10)
        batch.set_node_attr(0, "seed", "new")
        batch.set_node_attr(1, "seed", "new")
        batch.set_edge_attr(0, "length", -1)
        graph.apply_batch(batch, graph.current_state_id, "correction", "test")
        assert captured.label_propagation(seed_attr="seed") == [(0, "old"), (1, "old")]
    with gr.DurableGraph.open(path) as graph:
        assert graph.read_view_at(15).label_propagation(seed_attr="seed") == [(0, "new"), (1, "new")]
        with pytest.raises((ValueError, RuntimeError)):
            graph.read_view_at(15).girvan_newman(weight_attr="length")
    for method in ALL:
        assert run(captured, method) == expected[method]
