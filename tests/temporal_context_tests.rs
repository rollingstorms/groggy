use groggy::algorithms::{ChangedEntities, Context, TemporalDelta, TemporalScope};
use groggy::{AttrValue, Graph};

#[test]
fn test_context_with_temporal_scope() {
    // Create context with temporal scope
    let scope = TemporalScope::at_commit(42);
    let ctx = Context::with_temporal_scope(scope);

    assert!(ctx.temporal_scope().is_some());
    assert_eq!(ctx.temporal_scope().unwrap().current_commit, 42);
}

#[test]
fn test_context_temporal_scope_setter() {
    let mut ctx = Context::new();
    assert!(ctx.temporal_scope().is_none());

    let scope = TemporalScope::with_window(50, 10, 50);
    ctx.set_temporal_scope(scope);

    assert!(ctx.temporal_scope().is_some());
    assert_eq!(ctx.temporal_scope().unwrap().current_commit, 50);
    assert!(ctx.temporal_scope().unwrap().has_window());

    ctx.clear_temporal_scope();
    assert!(ctx.temporal_scope().is_none());
}

#[test]
fn test_context_delta_computation() {
    let mut graph = Graph::new();

    // Create first snapshot
    let node_a = graph.add_node();
    let node_b = graph.add_node();
    graph.add_edge(node_a, node_b).unwrap();
    let commit1 = graph.commit("v1".to_string(), "test".to_string()).unwrap();
    let snapshot1 = graph.snapshot_at_commit(commit1).unwrap();

    // Create second snapshot with changes
    let node_c = graph.add_node();
    graph.add_edge(node_b, node_c).unwrap();
    let commit2 = graph.commit("v2".to_string(), "test".to_string()).unwrap();
    let snapshot2 = graph.snapshot_at_commit(commit2).unwrap();

    // Use context to compute delta
    let ctx = Context::new();
    let delta = ctx.delta(&snapshot1, &snapshot2).unwrap();

    assert_eq!(delta.from_commit, commit1);
    assert_eq!(delta.to_commit, commit2);
    assert_eq!(delta.nodes_added.len(), 1);
    assert!(delta.nodes_added.contains(&node_c));
    assert_eq!(delta.nodes_removed.len(), 0);
    assert!(!delta.edges_added.is_empty()); // At least the new edge
}

#[test]
fn test_context_changed_entities() {
    let mut graph = Graph::new();

    let node_a = graph.add_node();
    let node_b = graph.add_node();
    graph.add_edge(node_a, node_b).unwrap();
    let commit1 = graph.commit("v1".to_string(), "test".to_string()).unwrap();

    // Make changes
    graph
        .set_node_attr(
            node_a,
            "status".to_string(),
            AttrValue::Text("active".into()),
        )
        .unwrap();
    let _commit2 = graph.commit("v2".to_string(), "test".to_string()).unwrap();

    let node_c = graph.add_node();
    let commit3 = graph.commit("v3".to_string(), "test".to_string()).unwrap();

    // Build temporal index
    let index = graph.build_temporal_index().unwrap();

    // Create context with window
    let mut ctx = Context::new();
    let scope = TemporalScope::with_window(commit3, commit1, commit3);
    ctx.set_temporal_scope(scope);

    // Get changed entities within window
    let changed = ctx.changed_entities(&index).unwrap();

    assert!(!changed.is_empty());
    assert!(changed.total_changes() > 0);
    // Node C was created in this window
    assert!(changed.modified_nodes.contains(&node_c));
}

#[test]
fn test_temporal_delta_summary() {
    let mut graph = Graph::new();

    let node_a = graph.add_node();
    let commit1 = graph.commit("v1".to_string(), "test".to_string()).unwrap();
    let snapshot1 = graph.snapshot_at_commit(commit1).unwrap();

    let node_b = graph.add_node();
    let node_c = graph.add_node();
    graph.add_edge(node_a, node_b).unwrap();
    graph.add_edge(node_b, node_c).unwrap();
    let commit2 = graph.commit("v2".to_string(), "test".to_string()).unwrap();
    let snapshot2 = graph.snapshot_at_commit(commit2).unwrap();

    let delta = TemporalDelta::compute(&snapshot1, &snapshot2).unwrap();
    let summary = delta.summary();

    // Summary should mention nodes and edges added
    assert!(summary.contains("nodes"));
    assert!(summary.contains("edges"));
    assert!(!delta.is_empty());
}

#[test]
fn test_temporal_scope_with_metadata() {
    let scope = TemporalScope::at_commit(10)
        .with_metadata("analysis_type".to_string(), "drift".to_string())
        .with_metadata("user".to_string(), "analyst".to_string());

    assert_eq!(
        scope.metadata.tags.get("analysis_type"),
        Some(&"drift".to_string())
    );
    assert_eq!(
        scope.metadata.tags.get("user"),
        Some(&"analyst".to_string())
    );
}

#[test]
fn test_changed_entities_merge() {
    let mut entities1 = ChangedEntities::empty();
    entities1.add_node(1, groggy::algorithms::ChangeType::Created);
    entities1.add_edge(100, groggy::algorithms::ChangeType::Created);

    let mut entities2 = ChangedEntities::empty();
    entities2.add_node(2, groggy::algorithms::ChangeType::AttributeModified);
    entities2.add_edge(200, groggy::algorithms::ChangeType::Deleted);

    entities1.merge(entities2);

    assert_eq!(entities1.total_changes(), 4);
    assert!(entities1.modified_nodes.contains(&1));
    assert!(entities1.modified_nodes.contains(&2));
    assert!(entities1.modified_edges.contains(&100));
    assert!(entities1.modified_edges.contains(&200));
}

#[test]
fn test_delta_affected_entities() {
    let mut graph = Graph::new();

    let node_a = graph.add_node();
    let node_b = graph.add_node();
    let edge_ab = graph.add_edge(node_a, node_b).unwrap();
    let commit1 = graph.commit("v1".to_string(), "test".to_string()).unwrap();
    let snapshot1 = graph.snapshot_at_commit(commit1).unwrap();

    let node_c = graph.add_node();
    let edge_bc = graph.add_edge(node_b, node_c).unwrap();
    let commit2 = graph.commit("v2".to_string(), "test".to_string()).unwrap();
    let snapshot2 = graph.snapshot_at_commit(commit2).unwrap();

    let delta = TemporalDelta::compute(&snapshot1, &snapshot2).unwrap();

    let affected_nodes = delta.affected_nodes();
    assert!(affected_nodes.contains(&node_c)); // New node

    let affected_edges = delta.affected_edges();
    assert!(affected_edges.contains(&edge_bc)); // New edge
    assert!(!affected_edges.contains(&edge_ab)); // Unchanged edge
}
