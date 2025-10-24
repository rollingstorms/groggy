import groggy as gr


def test_temporal_snapshot_python_api():
    g = gr.Graph()
    a = g.add_node()
    b = g.add_node()
    edge = g.add_edge(a, b)

    g.set_node_attr(a, "label", "A")
    g.set_node_attr(b, "label", "B")
    g.set_edge_attr(edge, "weight", 1.5)

    commit_id = g.commit("initial", "tests")

    # Make additional changes to ensure snapshot isolates the first commit
    c = g.add_node()
    g.add_edge(b, c)

    snapshot = g.snapshot_at_commit(commit_id)
    assert snapshot.commit_id == commit_id
    assert snapshot.node_exists(a)
    assert not snapshot.node_exists(c)
    assert snapshot.edge_exists(edge)

    neighbors = snapshot.neighbors(a)
    assert neighbors == [b]

    subgraph = snapshot.as_subgraph()
    assert subgraph.node_count() == 2

    timestamp_snapshot = g.snapshot_at_timestamp(snapshot.timestamp)
    assert timestamp_snapshot.commit_id == commit_id
