"""Manual and publishing graph copies validate before changing destination state."""
import pytest
from groggy import Graph


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("directed", [False, True])
def test_copy_preserves_active_ids_values_and_one_operation_boundary(manual, directed):
    source = Graph(directed=directed, manual=True)
    source.add_nodes([{}, {"v": 1.0000000000000002}, {"v": None}, {"v": b"bytes"}])
    source.remove_node(0)
    source.add_edges([(1, 2, {"v": [1.0000000000000002]}), (1, 2), (2, 2)])
    target = Graph(directed=directed, manual=manual)
    target.add_nodes(3)
    target.add_edge(0, 0)
    target.remove_node(2)
    before = target.commit("before", "test")
    target.add_graph(source)
    if not manual:
        assert target.current_state_id == before + 1
        assert not target.has_uncommitted_changes()
    after = target.commit("after", "test")
    assert set(target.node_ids) == {0, 1, 3, 4, 5}
    assert target.get_node_attr(3, "v") == 1.0000000000000002
    assert target.get_node_attr(5, "v") == b"bytes"
    assert target.snapshot_at_commit(after).node_attr(4, "v").type_name == "null"
    assert target.get_edge_attr(1, "v") == [1.0000000000000002]
    assert [target.edge_endpoints(i) for i in [1, 2, 3]] == [(3, 4), (3, 4), (4, 4)]
    assert source.has_uncommitted_changes()


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("identity_on_edge", [False, True])
def test_managed_identity_copy_failure_leaves_ids_and_history_unchanged(manual, identity_on_edge):
    source = Graph()
    if identity_on_edge:
        source.add_nodes(2)
        edge = source.add_edge(0, 1)
        source.set_edge_external_id(edge, "managed")
    else:
        source.upsert_nodes([{"external_id": "managed"}])
    target = Graph(manual=manual)
    with pytest.raises(ValueError, match="explicit upserts"):
        target.add_graph(source)
    assert target.current_state_id == 0
    assert not target.has_uncommitted_changes()
    assert target.add_node() == 0


@pytest.mark.parametrize("manual", [False, True])
def test_meta_copy_cannot_silently_become_base_nodes(manual):
    source = Graph()
    source.add_node()
    source.view().collapse()
    target = Graph(manual=manual)
    with pytest.raises(ValueError, match="pooled/meta"):
        target.add_graph(source)
    assert not target.has_uncommitted_changes()
    assert target.add_node() == 0
