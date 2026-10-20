use groggy::{AttrValue, BatchEdgeRef, BatchNodeRef, Graph, GraphBatch};

#[test]
fn mixed_batch_publishes_one_state_with_stable_prior_snapshot() {
    let mut graph = Graph::new();
    let mut first = GraphBatch {
        add_nodes: 2,
        add_edges: vec![(BatchNodeRef::Added(0), BatchNodeRef::Added(1))],
        ..Default::default()
    };
    first.node_attrs.insert(
        "name".into(),
        vec![(BatchNodeRef::Added(0), AttrValue::Text("Ada".into()))],
    );
    first.edge_attrs.insert(
        "weight".into(),
        vec![(BatchEdgeRef::Added(0), AttrValue::Float(1.5))],
    );
    let first_result = graph
        .apply_batch(first, 0, "first".into(), "test".into())
        .unwrap();
    assert_eq!(first_result.added_nodes, vec![0, 1]);
    assert_eq!(first_result.added_edges, vec![0]);
    assert_eq!(graph.current_state_id(), first_result.state_id);
    assert_eq!(graph.commit_history().len(), 1);
    let prior = graph.snapshot_at_commit(first_result.state_id).unwrap();

    let mut second = GraphBatch::default();
    second.node_attrs.insert(
        "name".into(),
        vec![(BatchNodeRef::Existing(0), AttrValue::Text("Grace".into()))],
    );
    second.remove_edges.push(BatchEdgeRef::Existing(0));
    let second_result = graph
        .apply_batch(
            second,
            first_result.state_id,
            "second".into(),
            "test".into(),
        )
        .unwrap();

    assert_eq!(graph.commit_history().len(), 2);
    assert_ne!(first_result.state_id, second_result.state_id);
    assert_eq!(
        prior.node_attr(0, &"name".into()),
        Some(AttrValue::Text("Ada".into()))
    );
    assert!(prior.edge_exists(0));
    let current = graph.snapshot_at_commit(second_result.state_id).unwrap();
    assert_eq!(
        current.node_attr(0, &"name".into()),
        Some(AttrValue::Text("Grace".into()))
    );
    assert!(!current.edge_exists(0));
}

#[test]
fn invalid_mixed_batch_reports_all_errors_without_allocation() {
    let mut graph = Graph::new();
    let mut batch = GraphBatch {
        add_nodes: 1,
        add_edges: vec![(BatchNodeRef::Added(0), BatchNodeRef::Existing(99))],
        ..Default::default()
    };
    batch.node_attrs.insert(
        "score".into(),
        vec![(BatchNodeRef::Existing(100), AttrValue::Int(1))],
    );
    batch.edge_attrs.insert(
        "weight".into(),
        vec![(BatchEdgeRef::Added(1), AttrValue::Int(2))],
    );
    batch.remove_nodes.push(BatchNodeRef::Existing(101));

    let error = graph
        .apply_batch(batch, 0, "bad".into(), "test".into())
        .unwrap_err()
        .to_string();
    for part in [
        "add_edges[0]",
        "node_attrs.score[0]",
        "edge_attrs.weight[0]",
        "remove_nodes[0]",
    ] {
        assert!(error.contains(part), "missing {part}: {error}");
    }
    assert_eq!(graph.current_state_id(), 0);
    assert!(graph.node_ids().is_empty());
    assert!(graph.edge_ids().is_empty());
    assert_eq!(graph.add_node(), 0);
}

#[test]
fn batch_rejects_stale_or_dirty_head_without_changes() {
    let mut graph = Graph::new();
    let first = graph
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                ..Default::default()
            },
            0,
            "first".into(),
            "test".into(),
        )
        .unwrap();
    let stale = graph.apply_batch(
        GraphBatch {
            add_nodes: 1,
            ..Default::default()
        },
        0,
        "stale".into(),
        "test".into(),
    );
    assert!(stale.unwrap_err().to_string().contains("Stale"));
    assert_eq!(graph.node_ids(), vec![0]);

    graph.add_node();
    let dirty = graph.apply_batch(
        GraphBatch {
            add_nodes: 1,
            ..Default::default()
        },
        first.state_id,
        "dirty".into(),
        "test".into(),
    );
    assert!(dirty.unwrap_err().to_string().contains("uncommitted"));
    let mut live_nodes = graph.node_ids();
    live_nodes.sort_unstable();
    assert_eq!(live_nodes, vec![0, 1]);
    assert_eq!(graph.current_state_id(), first.state_id);
}

#[test]
fn batch_rejects_edge_whose_endpoint_is_removed_in_same_write() {
    let mut graph = Graph::new();
    let mut batch = GraphBatch {
        add_nodes: 2,
        add_edges: vec![(BatchNodeRef::Added(0), BatchNodeRef::Added(1))],
        ..Default::default()
    };
    batch.remove_nodes.push(BatchNodeRef::Added(1));
    let error = graph
        .apply_batch(batch, 0, "contradictory".into(), "test".into())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("add_edges[0]: endpoint is removed"),
        "{error}"
    );
    assert!(graph.node_ids().is_empty());
    assert!(graph.edge_ids().is_empty());
    assert_eq!(graph.add_node(), 0);
}
