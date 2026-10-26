use groggy::{
    AttrValue, BatchEdgeRef as E, BatchNodeRef as N, EdgeUpsert, Graph, GraphBatch, GraphSchema,
    NodeUpsert,
};
use std::collections::HashMap;
#[test]
fn mixed_identity_assignments_publish_data_and_index_together() {
    let mut graph = Graph::new_manual();
    let batch = GraphBatch {
        add_nodes: 2,
        add_edges: vec![(N::Added(0), N::Added(1))],
        node_external_ids: vec![(N::Added(0), "ada".into()), (N::Added(1), "lin".into())],
        edge_external_ids: vec![(E::Added(0), "fact".into())],
        node_attrs: HashMap::from([(
            "labels".into(),
            vec![(N::Added(0), AttrValue::TextVec(vec!["Entity".into()]))],
        )]),
        ..Default::default()
    };
    let schema =
        GraphSchema::from_json(r#"{"node_labels":{"Entity":{"external_id":{"type":"text"}}}}"#)
            .unwrap();
    let result = graph
        .apply_batch_with_schema(batch, 0, "mixed".into(), "test".into(), &schema)
        .unwrap();
    assert_eq!(graph.node_by_external_id("ada", None), Some(0));
    assert_eq!(graph.edge_by_external_id("fact", None), Some(0));
    let snapshot = graph.snapshot_at_commit(result.state_id).unwrap();
    let renamed = GraphBatch {
        node_external_ids: vec![(N::Existing(0), "ada-new".into())],
        ..Default::default()
    };
    graph
        .apply_batch(renamed, result.state_id, "rename".into(), "test".into())
        .unwrap();
    assert_eq!(snapshot.node_by_external_id("ada", None), Some(0));
    assert_eq!(graph.node_by_external_id("ada-new", None), Some(0));
    assert!(graph.allocated_entity_for_external_id("ada").is_some());
}
#[test]
fn collisions_and_schema_failures_do_not_reserve_names_or_consume_ids() {
    let mut graph = Graph::new_manual();
    let collision = GraphBatch {
        add_nodes: 1,
        add_edges: vec![(N::Added(0), N::Added(0))],
        node_external_ids: vec![(N::Added(0), "same".into())],
        edge_external_ids: vec![(E::Added(0), "same".into())],
        ..Default::default()
    };
    assert!(graph
        .apply_batch(collision, 0, "bad".into(), "test".into())
        .is_err());
    assert!(graph.allocated_entity_for_external_id("same").is_none());
    assert_eq!(graph.current_state_id(), 0);
    let schema =
        GraphSchema::from_json(r#"{"node_labels":{"Entity":{"name":{"type":"text"}}}}"#).unwrap();
    let bad = GraphBatch {
        add_nodes: 1,
        node_external_ids: vec![(N::Added(0), "unreserved".into())],
        node_attrs: HashMap::from([(
            "labels".into(),
            vec![(N::Added(0), AttrValue::TextVec(vec!["Entity".into()]))],
        )]),
        ..Default::default()
    };
    assert!(graph
        .apply_batch_with_schema(bad, 0, "bad schema".into(), "test".into(), &schema)
        .is_err());
    assert!(graph
        .allocated_entity_for_external_id("unreserved")
        .is_none());
    assert_eq!(graph.add_node().unwrap(), 0);
}
#[test]
fn writer_upserts_preserve_identity_and_do_not_publish_identical_payloads() {
    let mut graph = Graph::new_manual();
    let mut writer = graph.writer("test");
    let row = NodeUpsert {
        external_id: "ada".into(),
        attrs: HashMap::from([("name".into(), AttrValue::Text("Ada".into()))]),
    };
    assert_eq!(writer.upsert_nodes(vec![row.clone()]).unwrap(), vec![0]);
    let first = writer.state_id();
    assert_eq!(writer.upsert_nodes(vec![row]).unwrap(), vec![0]);
    assert_eq!(writer.state_id(), first);
    let edge = EdgeUpsert {
        external_id: "fact".into(),
        source: 0,
        target: 0,
        attrs: HashMap::new(),
    };
    assert_eq!(writer.upsert_edges(vec![edge.clone()]).unwrap(), vec![0]);
    let second = writer.state_id();
    assert_eq!(writer.upsert_edges(vec![edge]).unwrap(), vec![0]);
    assert_eq!(writer.state_id(), second);
    assert_eq!(writer.graph().commit_history().len(), 2);
    assert!(!writer.graph().has_uncommitted_changes());
}
