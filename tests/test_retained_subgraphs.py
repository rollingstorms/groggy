import pytest

from groggy import Graph


@pytest.mark.parametrize("allow_missing", [False, True])
@pytest.mark.parametrize("strategy", ["extract", "collapse"])
def test_stored_membership_survives_collapse_deletion_and_branch_switch(allow_missing, strategy):
    graph = Graph()
    a = graph.add_node(name="Ada")
    b = graph.add_node(name="Lin")
    edge = graph.add_edge(a, b)
    graph.commit("original", "test")
    graph.create_branch("original")
    meta = graph.view().collapse(
        node_aggs={"size": "count"},
        node_strategy=strategy,
        allow_missing_attributes=allow_missing,
    )
    assert set(meta.expand().node_ids) == {a, b}
    assert set(meta.expand().edge_ids) == {edge}
    if strategy == "extract":
        assert set(graph.node_ids) == {a, b, meta.id}
        graph.remove_node(b)
    else:
        assert set(graph.node_ids) == {meta.id}
    graph.commit("group", "test")
    assert set(meta.expand().node_ids) == {a, b}
    assert set(meta.expand().edge_ids) == {edge}
    graph.checkout_branch("original")
    assert set(graph.node_ids) == {a, b}
    assert set(graph.edge_ids) == {edge}
    graph.checkout_branch("main")
    assert set(meta.expand().node_ids) == {a, b}
    assert set(meta.expand().edge_ids) == {edge}


def test_extracting_a_nested_group_preserves_meta_identity():
    graph = Graph()
    graph.add_node(name="Ada")
    inner = graph.view().collapse(node_aggs={"size": "count"}, node_strategy="extract")
    outer = graph.view().collapse(node_aggs={"size": "count"}, node_strategy="extract")
    assert inner.id in set(outer.expand().node_ids)
    assert graph.get_node_attr(inner.id, "entity_type") == "meta"
    assert graph.get_node_attr(outer.id, "entity_type") == "meta"
