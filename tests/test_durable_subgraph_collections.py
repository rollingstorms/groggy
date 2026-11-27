"""Portable collections retain immutable sources and fixed selected child frames."""

import pytest

from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def pipeline(steps):
    return Pipeline([step_pipeline("portable collections", steps, atomic=True)])


def fixture(path, directed=True):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=0)
    for _ in range(4):
        batch.add_node()
    for a, b in [(0, 1), (0, 1), (1, 1), (1, 2), (2, 0)]:
        batch.add_edge(a, b)
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


def capture():
    return [
        step("graph.capture_staged", output="pin"),
        step("graph.emit", source="pin", name="child"),
    ]


def increment():
    return [
        step("core.load_node_attr", attr="counter", default=0, target="old"),
        step("core.init_scalar", value=1, target="one"),
        step("core.add", left="old", right="one", target="next"),
        step("core.attach_node_attr", source="next", attr="counter", effective_at=20),
    ]


def child_views(profile):
    return [v for k, v in sorted(profile["outputs"].items()) if k.endswith(".child")]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("edge_seeds", [False, True])
@pytest.mark.parametrize("hops", [0, 1])
@pytest.mark.parametrize("unified", [False, True])
def test_collection_membership_overlapping_writes_nested_loops_and_immutable_outputs(
    tmp_path, directed, edge_seeds, hops, unified
):
    path = tmp_path / "collections.journal"
    store = fixture(path, directed)
    source = store.read_view().select_nodes([0, 1, 2])
    edges = {0: (2, 1), 2: (1, 1), 3: (1, 2), 4: (2, 0), 5: (4, 2)}
    nodes = [0, 1, 2, 4]
    selections = (
        [set(pair) for pair in edges.values()] if edge_seeds else [{n} for n in nodes]
    )
    expanded = []
    for seed in selections:
        selected = set(seed)
        if hops:
            for a, b in edges.values():
                if a in seed:
                    selected.add(b)
                if not directed and b in seed:
                    selected.add(a)
        expanded.append(selected)
    if unified:
        expanded = [set.union(*expanded)]
    totals = dict.fromkeys(nodes, 0)
    expected_counters = []
    for selected in expanded:
        for n in selected:
            totals[n] += 2
        expected_counters.append({n: totals[n] for n in selected})
    result, profile = pipeline(
        [
            step("graph.add_nodes", count=1, output="new", effective_at=10),
            step(
                "graph.add_edges",
                edges=[[{"added": 0}, 2]],
                output="edge",
                effective_at=10,
            ),
            step("graph.set_edge_endpoints", revisions=[[0, 2, 1]], effective_at=10),
            step("graph.remove_edges", ids=[1], effective_at=10),
            step(
                "sample.iterate_edges" if edge_seeds else "sample.iterate_nodes",
                target="seeds",
            ),
            step("alias", source="seeds", target="saved"),
            step("sample.neighbors", source="seeds", hops=hops, target="expanded"),
            step(
                "sample.emit_subgraphs",
                source="expanded",
                target="emitted",
                mode="unified" if unified else "per_item",
            ),
            step(
                "sample.for_each",
                source="emitted",
                target="mapped",
                body=[step("iter.loop", iterations=2, body=increment()), *capture()],
            ),
            step("sample.for_each", source="saved", target="preserved", body=[]),
        ]
    ).run_durable(store, source, return_profile=True)
    views = child_views(profile)
    assert len(views) == len(expanded)
    for view, selected, counts in zip(views, expanded, expected_counters):
        assert set(view.node_ids) == selected
        assert set(view.edge_ids) == {
            e for e, (a, b) in edges.items() if a in selected and b in selected
        }
        assert {n: view.node_attr(n, "counter") for n in selected} == counts
    assert {n: result.node_attr(n, "counter") for n in nodes} == totals
    assert source.node_attr(0, "counter") is None
    assert 3 not in result.node_ids
    store.close()
    recovered = DurableGraph.open(path)
    assert recovered.read_view_at(19).node_attr(0, "counter") is None
    assert {
        n: recovered.read_view_at(20).node_attr(n, "counter") for n in nodes
    } == totals
    assert views[0].node_attr(next(iter(expanded[0])), "counter") == 2
    recovered.close()


@pytest.mark.parametrize("kind", ["nodes", "edges"])
@pytest.mark.parametrize("induced", [False, True])
def test_map_emission_distinguishes_explicit_edge_scope_and_induction(
    tmp_path, kind, induced
):
    store = fixture(tmp_path / "maps.journal")
    _, profile = pipeline(
        [
            step(
                "core.init_edges" if kind == "edges" else "core.init_nodes",
                target="mask",
                value=True,
            ),
            step(
                "sample.emit_subgraphs",
                source="mask",
                target="seeds",
                mode="unified",
                induced=induced,
            ),
            step("sample.for_each", source="seeds", target="mapped", body=capture()),
        ]
    ).run_durable(store, return_profile=True)
    view = child_views(profile)[0]
    assert view.node_ids == ([0, 1, 2] if kind == "edges" else [0, 1, 2, 3])
    assert view.edge_ids == (list(range(5)) if kind == "edges" or induced else [])
    store.close()


@pytest.mark.parametrize("explicit_empty", [False, True])
def test_nested_foreach_distinguishes_explicit_empty_output_from_passthrough(
    tmp_path, explicit_empty
):
    store = fixture(tmp_path / "empty.journal")
    body = [step("sample.iterate_nodes", target="inner")]
    if explicit_empty:
        body += [
            step("core.sample_nodes", count=0, target="none"),
            step(
                "sample.emit_subgraphs",
                source="none",
                target="__sample_output__",
                mode="per_item",
            ),
        ]
    body += [step("sample.for_each", source="inner", target="mapped", body=increment())]
    result, profile = pipeline(
        [
            step("sample.iterate_nodes", target="seeds"),
            step("sample.for_each", source="seeds", target="outer", body=body),
            step("sample.for_each", source="outer", target="last", body=capture()),
        ]
    ).run_durable(store, return_profile=True)
    assert [result.node_attr(n, "counter") for n in range(4)] == [1] * 4
    assert len(child_views(profile)) == (0 if explicit_empty else 4)
    store.close()


@pytest.mark.parametrize(
    "body",
    [
        [step("core.init_nodes", target="__sample_output__", value=1)],
        [
            step(
                "iter.loop",
                iterations=0,
                body=[step("graph.add_nodes", count=1, output="new")],
            )
        ],
    ],
)
def test_bad_child_output_and_topology_reject_without_journal_or_id_consumption(
    tmp_path, body
):
    path = tmp_path / "rejected.journal"
    store = fixture(path)
    before, head = path.read_bytes(), store.current_state_id
    with pytest.raises(Exception):
        pipeline(
            [
                step("graph.add_nodes", count=1, output="new"),
                step("sample.iterate_nodes", target="seeds"),
                step("sample.for_each", source="seeds", target="mapped", body=body),
            ]
        ).run_durable(store)
    assert path.read_bytes() == before
    assert store.current_state_id == head
    assert store.read_view().node_ids == [0, 1, 2, 3]
    store.close()


@pytest.mark.parametrize("unified", [False, True])
def test_empty_array_and_empty_node_mask_emission_preserve_read_only_behavior(
    tmp_path, unified
):
    path = tmp_path / "empty-read.journal"
    store = fixture(path)
    head, before = store.current_state_id, path.read_bytes()
    _, profile = pipeline(
        [
            step("core.sample_nodes", count=0, target="mask"),
            step(
                "sample.emit_subgraphs",
                source="mask",
                target="seeds",
                mode="unified" if unified else "per_item",
            ),
            step("sample.for_each", source="seeds", target="mapped", body=capture()),
        ]
    ).run_durable(store, return_profile=True)
    views = child_views(profile)
    assert len(views) == (1 if unified else 0)
    if unified:
        assert views[0].node_ids == []
        assert views[0].edge_ids == []
    assert store.current_state_id == head
    assert path.read_bytes() == before
    store.close()


@pytest.mark.parametrize(
    "consumer", ["sample.neighbors", "sample.emit_subgraphs", "sample.for_each"]
)
def test_stale_selectors_reject_instead_of_silently_filtering_membership(
    tmp_path, consumer
):
    path = tmp_path / "stale.journal"
    store = fixture(path)
    before, head = path.read_bytes(), store.current_state_id
    params = {"source": "seeds", "target": "out"}
    if consumer == "sample.for_each":
        params["body"] = increment()
    with pytest.raises(Exception, match="outside"):
        pipeline(
            [
                step("sample.iterate_nodes", target="seeds"),
                step("graph.remove_nodes", ids=[1]),
                step(consumer, **params),
            ]
        ).run_durable(store)
    assert store.current_state_id == head
    assert path.read_bytes() == before
    store.close()
