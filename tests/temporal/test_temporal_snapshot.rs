use groggy::temporal::TemporalSnapshot;
use groggy::{AttrValue, Graph};

#[test]
fn snapshot_at_commit_produces_expected_view() {
    let mut graph = Graph::new();
    let a = graph.add_node();
    let b = graph.add_node();
    let edge = graph.add_edge(a, b).unwrap();

    graph
        .set_node_attr(a, "label".to_string(), AttrValue::Text("A".into()))
        .unwrap();
    graph
        .set_node_attr(b, "label".to_string(), AttrValue::Text("B".into()))
        .unwrap();
    graph
        .set_edge_attr(edge, "weight".to_string(), AttrValue::Float(1.5))
        .unwrap();

    let commit_id = graph
        .commit("initial".to_string(), "tests".to_string())
        .unwrap();

    // Apply further changes so snapshot is immutable view of first state
    let c = graph.add_node();
    let _ = graph.add_edge(b, c).unwrap();
    graph
        .set_node_attr(c, "label".to_string(), AttrValue::Text("C".into()))
        .unwrap();

    let snapshot = graph.snapshot_at_commit(commit_id).unwrap();
    assert_eq!(snapshot.lineage().commit_id, commit_id);
    assert!(snapshot.node_exists(a));
    assert!(snapshot.node_exists(b));
    assert!(!snapshot.node_exists(c));
    assert!(snapshot.edge_exists(edge));

    let label = snapshot
        .node_attr(a, &"label".into())
        .expect("label should exist");
    assert_eq!(label, AttrValue::Text("A".into()));

    let neighbors = snapshot.neighbors(a).unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0], b);

    // Temporal snapshot should produce a standalone subgraph with the same nodes
    let subgraph = snapshot.as_subgraph().unwrap();
    assert_eq!(subgraph.node_count(), 2);
    assert!(subgraph.has_node(a));
    assert!(subgraph.has_node(b));
}

#[test]
fn snapshot_at_timestamp_matches_commit() {
    let mut graph = Graph::new();
    let first = graph.add_node();
    let second = graph.add_node();
    let _ = graph.add_edge(first, second).unwrap();

    let commit_id = graph
        .commit("initial".to_string(), "tests".to_string())
        .unwrap();

    let snapshot_commit = graph.snapshot_at_commit(commit_id).unwrap();
    let timestamp = snapshot_commit.lineage().timestamp;

    let snapshot_time = graph.snapshot_at_timestamp(timestamp).unwrap();
    assert_eq!(snapshot_time.lineage().commit_id, commit_id);
    assert!(snapshot_time.node_exists(first));
    assert!(snapshot_time.node_exists(second));
}
