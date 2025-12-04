import pytest

import groggy as gr


@pytest.mark.skip(reason="TemporalScope not yet implemented")
def test_temporal_scope_creation():
    """Test creating temporal scopes."""
    # Simple scope at a commit
    scope = gr.TemporalScope(current_commit=42, window=None)
    assert scope.current_commit == 42
    assert not scope.has_window()

    # Scope with a window
    scope_with_window = gr.TemporalScope(current_commit=50, window=(10, 50))
    assert scope_with_window.current_commit == 50
    assert scope_with_window.has_window()
    assert scope_with_window.window == (10, 50)
    assert scope_with_window.window_size() == 40


def test_temporal_delta_computation():
    """Test computing deltas between snapshots."""
    g = gr.Graph()

    # Create first snapshot
    node_a = g.add_node()
    node_b = g.add_node()
    g.add_edge(node_a, node_b)
    commit1 = g.commit("v1", "test")
    snapshot1 = g.snapshot_at_commit(commit1)

    # Create second snapshot with changes
    node_c = g.add_node()
    g.add_edge(node_b, node_c)
    commit2 = g.commit("v2", "test")
    snapshot2 = g.snapshot_at_commit(commit2)

    # Compute delta
    index = g.build_temporal_index()

    # Note: We can't directly call TemporalDelta.compute from Python
    # without exposing it, but we can test the result indirectly
    # through the index's ability to track changes

    nodes_at_c1 = index.nodes_at_commit(commit1)
    nodes_at_c2 = index.nodes_at_commit(commit2)

    assert len(nodes_at_c2) > len(nodes_at_c1)
    assert node_c in nodes_at_c2
    assert node_c not in nodes_at_c1


def test_changed_entities_from_index():
    """Test tracking changed entities using the temporal index."""
    g = gr.Graph()

    node_a = g.add_node()
    node_b = g.add_node()
    g.add_edge(node_a, node_b)
    commit1 = g.commit("v1", "test")

    # Make changes
    g.set_node_attr(node_a, "status", "active")
    commit2 = g.commit("v2", "test")

    node_c = g.add_node()
    commit3 = g.commit("v3", "test")

    # Build temporal index
    index = g.build_temporal_index()

    # Track what changed in each commit
    changed_in_c1 = index.nodes_changed_in_commit(commit1)
    assert len(changed_in_c1) == 2  # Both nodes were created

    changed_in_c2 = index.nodes_changed_in_commit(commit2)
    assert len(changed_in_c2) == 1  # Only node A changed
    assert node_a in changed_in_c2

    changed_in_c3 = index.nodes_changed_in_commit(commit3)
    assert len(changed_in_c3) == 1  # Node C was created
    assert node_c in changed_in_c3


@pytest.mark.skip(reason="TemporalScope not yet implemented")
def test_temporal_scope_metadata():
    """Test adding metadata to temporal scopes."""
    scope = gr.TemporalScope(current_commit=10, window=None)
    # Note: with_metadata returns a new scope, so we need to chain calls
    # However, the current implementation doesn't expose a Python-friendly API for this
    # This is noted for future improvement
    assert scope.current_commit == 10


def test_temporal_window_queries():
    """Test querying within temporal windows."""
    g = gr.Graph()

    node_a = g.add_node()
    commit1 = g.commit("c1", "test")

    node_b = g.add_node()
    g.add_edge(node_a, node_b)
    commit2 = g.commit("c2", "test")

    node_c = g.add_node()
    commit3 = g.commit("c3", "test")

    # Build index
    index = g.build_temporal_index()

    # Query nodes in window
    nodes_c1 = index.nodes_at_commit(commit1)
    nodes_c2 = index.nodes_at_commit(commit2)
    nodes_c3 = index.nodes_at_commit(commit3)

    assert len(nodes_c1) == 1  # Only node A
    assert len(nodes_c2) == 2  # A and B
    assert len(nodes_c3) == 3  # A, B, and C

    # Test neighbor queries across time
    neighbors_window = index.neighbors_in_window(node_a, commit1, commit3)
    assert node_b in neighbors_window  # B was connected to A during window


def test_temporal_index_with_attribute_history():
    """Test tracking attribute changes over time."""
    g = gr.Graph()

    node_a = g.add_node()
    g.set_node_attr(node_a, "status", "initial")
    commit1 = g.commit("c1", "test")

    g.set_node_attr(node_a, "status", "active")
    commit2 = g.commit("c2", "test")

    g.set_node_attr(node_a, "status", "inactive")
    commit3 = g.commit("c3", "test")

    # Build index
    index = g.build_temporal_index()

    # Get attribute history
    history = index.node_attr_history(node_a, "status", commit1, commit3)
    assert len(history) == 3  # Three changes

    # Each entry is (commit_id, value)
    commit_ids = [h[0] for h in history]
    assert commit1 in commit_ids
    assert commit2 in commit_ids
    assert commit3 in commit_ids


def test_temporal_edge_tracking():
    """Test tracking edge changes over time."""
    g = gr.Graph()

    node_a = g.add_node()
    node_b = g.add_node()
    commit1 = g.commit("nodes only", "test")

    edge_ab = g.add_edge(node_a, node_b)
    commit2 = g.commit("with edge", "test")

    # Build index
    index = g.build_temporal_index()

    # Check edge existence at different commits
    assert not index.edge_exists_at(edge_ab, commit1)
    assert index.edge_exists_at(edge_ab, commit2)

    # Get edges at each commit
    edges_c1 = index.edges_at_commit(commit1)
    edges_c2 = index.edges_at_commit(commit2)

    assert len(edges_c1) == 0
    assert len(edges_c2) == 1
    assert edge_ab in edges_c2


def test_bulk_temporal_neighbor_queries():
    """Test bulk neighbor queries at specific commits."""
    g = gr.Graph()

    node_a = g.add_node()
    node_b = g.add_node()
    node_c = g.add_node()

    g.add_edge(node_a, node_b)
    commit1 = g.commit("partial graph", "test")

    g.add_edge(node_b, node_c)
    g.add_edge(node_c, node_a)
    commit2 = g.commit("complete graph", "test")

    # Build index
    index = g.build_temporal_index()

    # Bulk query at commit1
    bulk_c1 = index.neighbors_bulk_at_commit([node_a, node_b, node_c], commit1)
    assert len(bulk_c1[node_a]) == 1  # A connects to B
    assert len(bulk_c1[node_b]) == 1  # B connects to A
    assert len(bulk_c1[node_c]) == 0  # C has no edges yet

    # Bulk query at commit2
    bulk_c2 = index.neighbors_bulk_at_commit([node_a, node_b, node_c], commit2)
    assert len(bulk_c2[node_a]) == 2  # A connects to B and C
    assert len(bulk_c2[node_b]) == 2  # B connects to A and C
    assert len(bulk_c2[node_c]) == 2  # C connects to B and A


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
