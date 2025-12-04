import pytest

import groggy as gr


def test_temporal_index_basic():
    """Test basic temporal index functionality."""
    g = gr.Graph()

    # Create initial graph state
    node_a = g.add_node()
    node_b = g.add_node()
    edge_ab = g.add_edge(node_a, node_b)

    g.set_node_attr(node_a, "label", "A")
    g.set_node_attr(node_b, "label", "B")

    commit1 = g.commit("Initial graph", "tester")

    # Add more nodes
    node_c = g.add_node()
    g.set_node_attr(node_c, "label", "C")
    g.set_node_attr(node_a, "status", "active")
    edge_bc = g.add_edge(node_b, node_c)

    commit2 = g.commit("Added node C", "tester")

    # Modify attribute
    g.set_node_attr(node_a, "status", "inactive")
    commit3 = g.commit("Changed status", "tester")

    # Build temporal index
    index = g.build_temporal_index()

    # Test node existence at different commits
    assert index.node_exists_at(node_a, commit1)
    assert index.node_exists_at(node_b, commit1)
    assert not index.node_exists_at(node_c, commit1)  # C doesn't exist yet

    assert index.node_exists_at(node_c, commit2)  # C now exists
    assert index.node_exists_at(node_c, commit3)

    # Test edge existence
    assert index.edge_exists_at(edge_ab, commit1)
    assert index.edge_exists_at(edge_ab, commit2)
    assert index.edge_exists_at(edge_ab, commit3)

    # Test neighbor queries
    neighbors_commit1 = index.neighbors_at_commit(node_b, commit1)
    assert len(neighbors_commit1) == 1
    assert node_a in neighbors_commit1

    neighbors_commit2 = index.neighbors_at_commit(node_b, commit2)
    assert len(neighbors_commit2) == 2  # Connected to both A and C
    assert node_a in neighbors_commit2
    assert node_c in neighbors_commit2

    # Test bulk neighbor query
    bulk_neighbors = index.neighbors_bulk_at_commit([node_a, node_b, node_c], commit2)
    assert len(bulk_neighbors) == 3
    assert len(bulk_neighbors[node_a]) == 1  # A connects to B
    assert len(bulk_neighbors[node_b]) == 2  # B connects to A and C
    assert len(bulk_neighbors[node_c]) == 1  # C connects to B

    # Test window queries
    neighbors_in_window = index.neighbors_in_window(node_b, commit1, commit3)
    assert node_a in neighbors_in_window
    assert node_c in neighbors_in_window

    # Test statistics
    stats = index.statistics()
    assert stats.total_nodes == 3
    assert stats.total_edges >= 2
    assert stats.total_commits == 3


def test_nodes_changed_in_commit():
    """Test tracking which nodes changed in each commit."""
    g = gr.Graph()

    node_a = g.add_node()
    node_b = g.add_node()
    edge_ab = g.add_edge(node_a, node_b)

    commit1 = g.commit("Initial", "test")

    # Modify only node A
    g.set_node_attr(node_a, "changed", 1)
    commit2 = g.commit("Changed A", "test")

    index = g.build_temporal_index()

    changed_commit1 = index.nodes_changed_in_commit(commit1)
    assert len(changed_commit1) == 2  # Both nodes were created

    changed_commit2 = index.nodes_changed_in_commit(commit2)
    assert len(changed_commit2) == 1  # Only node A changed
    assert node_a in changed_commit2


def test_graph_api_temporal_methods():
    """Test temporal methods on the Graph API."""
    g = gr.Graph()

    node_a = g.add_node()
    node_b = g.add_node()
    g.add_edge(node_a, node_b)

    g.set_node_attr(node_a, "status", "v1")
    commit1 = g.commit("v1", "test")

    g.set_node_attr(node_a, "status", "v2")
    commit2 = g.commit("v2", "test")

    # Test Graph API methods
    neighbors = g.neighbors_at_commit([node_a], commit1)
    assert node_a in neighbors
    assert len(neighbors[node_a]) == 1

    neighbors_window = g.neighbors_in_window(node_a, commit1, commit2)
    assert len(neighbors_window) == 1

    history = g.node_attr_history(node_a, "status", commit1, commit2)
    assert len(history) == 2  # Two changes
    # Each entry is (commit_id, value) and should be in chronological order
    commit_ids = [h[0] for h in history]
    assert commit1 in commit_ids
    assert commit2 in commit_ids


def test_temporal_index_nodes_and_edges_at_commit():
    """Test getting all nodes/edges that existed at a commit."""
    g = gr.Graph()

    node_a = g.add_node()
    node_b = g.add_node()
    edge_ab = g.add_edge(node_a, node_b)
    commit1 = g.commit("Two nodes", "test")

    node_c = g.add_node()
    edge_bc = g.add_edge(node_b, node_c)
    commit2 = g.commit("Three nodes", "test")

    index = g.build_temporal_index()

    # At commit1, only A and B exist
    nodes_at_commit1 = index.nodes_at_commit(commit1)
    assert len(nodes_at_commit1) == 2
    assert node_a in nodes_at_commit1
    assert node_b in nodes_at_commit1
    assert node_c not in nodes_at_commit1

    # At commit2, all three nodes exist
    nodes_at_commit2 = index.nodes_at_commit(commit2)
    assert len(nodes_at_commit2) == 3
    assert node_a in nodes_at_commit2
    assert node_b in nodes_at_commit2
    assert node_c in nodes_at_commit2

    # Test edges
    edges_at_commit1 = index.edges_at_commit(commit1)
    assert len(edges_at_commit1) == 1
    assert edge_ab in edges_at_commit1

    edges_at_commit2 = index.edges_at_commit(commit2)
    assert len(edges_at_commit2) == 2
    assert edge_ab in edges_at_commit2
    assert edge_bc in edges_at_commit2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
