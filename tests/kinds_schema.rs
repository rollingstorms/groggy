use groggy::{
    AttrValue, BatchNodeRef, EdgeUpsert, ExternalEntityRef, Graph, GraphBatch, GraphSchema,
    NodeUpsert,
};
use std::collections::HashMap;
fn labels(names: &[&str]) -> AttrValue {
    AttrValue::TextVec(names.iter().map(|s| (*s).into()).collect())
}

#[test]
fn kinds_preserve_storage_identity_parallel_edges_and_snapshots() {
    let mut graph = Graph::new_directed();
    let node = graph
        .add_nodes_with_attrs(vec![HashMap::from([(
            "labels".into(),
            labels(&["Person", "Entity"]),
        )])])
        .unwrap()[0];
    let first = graph.add_edge(node, node).unwrap();
    let second = graph.add_edge(node, node).unwrap();
    graph
        .set_edge_relationship_type(first, Some("KNOWS".into()))
        .unwrap();
    graph
        .set_edge_relationship_type(second, Some("KNOWS".into()))
        .unwrap();
    assert_eq!(graph.node_labels(node).unwrap(), vec!["Entity", "Person"]);
    assert_eq!(
        graph
            .get_node_attr(node, &"entity_type".into())
            .unwrap()
            .unwrap()
            .get_text()
            .as_deref(),
        Some("base")
    );
    assert_eq!(graph.nodes_with_label("Entity").unwrap(), vec![node]);
    assert_eq!(
        graph.edges_with_relationship_type("KNOWS").unwrap(),
        vec![first, second]
    );
    let state = graph.commit("kinds".into(), "test".into()).unwrap();
    let snapshot = graph.snapshot_at_commit(state).unwrap();
    graph.set_node_labels(node, vec![]).unwrap();
    graph.set_edge_relationship_type(first, None).unwrap();
    assert_eq!(
        snapshot.node_attr(node, &"labels".into()),
        Some(labels(&["Entity", "Person"]))
    );
    assert_eq!(
        snapshot
            .edge_attr(first, &"relationship_type".into())
            .unwrap()
            .get_text()
            .as_deref(),
        Some("KNOWS")
    );
    assert_eq!(
        graph.edges_with_relationship_type("KNOWS").unwrap(),
        vec![second]
    );
    graph.remove_node(node).unwrap();
    assert!(graph.nodes_with_label("Person").unwrap().is_empty());
    assert!(graph.node_labels(node).is_err());
    assert!(graph.edge_relationship_type(second).is_err());
}

#[test]
fn rejected_bulk_column_creation_and_batch_kind_values_leave_no_writes() {
    let mut graph = Graph::new();
    assert!(graph
        .add_nodes_with_attrs(vec![
            HashMap::new(),
            HashMap::from([("labels".into(), labels(&["same", "same"]))])
        ])
        .is_err());
    assert_eq!(graph.add_node(), 0);
    graph.commit("first".into(), "test".into()).unwrap();
    let values = HashMap::from([
        ("score".into(), vec![(0, AttrValue::Int(1))]),
        ("labels".into(), vec![(0, AttrValue::Text("bad".into()))]),
    ]);
    assert!(graph.set_node_attrs(values).is_err());
    assert!(graph.get_node_attr(0, &"score".into()).unwrap().is_none());
    assert!(graph
        .set_node_attr_column("labels".into(), vec![(0, labels(&[" "]))])
        .is_err());
    assert!(!graph.has_uncommitted_changes());
    let batch = GraphBatch {
        add_nodes: 1,
        node_attrs: HashMap::from([(
            "labels".into(),
            vec![(BatchNodeRef::Added(0), AttrValue::Null)],
        )]),
        ..Default::default()
    };
    assert!(graph
        .apply_batch(batch, graph.current_state_id(), "bad".into(), "test".into())
        .is_err());
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(graph.add_node(), 1);
}

#[test]
fn upserts_validate_kinds_before_allocation_and_normalize_idempotently() {
    let mut graph = Graph::new();
    let row = NodeUpsert {
        external_id: "a".into(),
        attrs: HashMap::from([("labels".into(), labels(&["Person", "Entity"]))]),
    };
    let id = graph.upsert_nodes(vec![row.clone()]).unwrap()[0];
    graph.commit("first".into(), "test".into()).unwrap();
    assert_eq!(graph.upsert_nodes(vec![row]).unwrap(), vec![id]);
    assert!(!graph.has_uncommitted_changes());
    assert!(graph
        .upsert_nodes(vec![NodeUpsert {
            external_id: "bad".into(),
            attrs: HashMap::from([("labels".into(), labels(&[""]))])
        }])
        .is_err());
    assert!(graph
        .upsert_edges(vec![EdgeUpsert {
            external_id: "bad-edge".into(),
            source: id,
            target: id,
            attrs: HashMap::from([("relationship_type".into(), AttrValue::Int(1))])
        }])
        .is_err());
    assert_eq!(graph.add_node(), 1);
    assert_eq!(graph.add_edge(id, id).unwrap(), 0);
}

#[test]
fn schema_audit_reports_presence_null_and_logical_types_without_mutating() {
    let mut graph = Graph::new();
    let node = graph
        .add_nodes_with_attrs(vec![HashMap::from([
            ("labels".into(), labels(&["Person", "Entity"])),
            ("age".into(), AttrValue::Null),
            ("extra".into(), AttrValue::Bool(true)),
        ])])
        .unwrap()[0];
    let edge = graph.add_edge(node, node).unwrap();
    graph
        .set_edge_relationship_type(edge, Some("KNOWS".into()))
        .unwrap();
    graph
        .set_edge_attr(edge, "weight".into(), AttrValue::Text("bad".into()))
        .unwrap();
    graph.commit("fixture".into(), "test".into()).unwrap();
    let schema=GraphSchema::from_json(r#"{"node_labels":{"Person":{"name":{"type":"text"},"age":{"type":"int","required":false}},"Entity":{"id":{"type":"int","required":false,"nullable":true}}},"relationship_types":{"KNOWS":{"weight":{"type":"float"}}}}"#).unwrap();
    let issues = graph.validate_schema(&schema).unwrap();
    assert_eq!(issues.len(), 3);
    assert_eq!(issues[0].property, "age");
    assert_eq!(issues[0].entity, ExternalEntityRef::Node(node));
    assert_eq!(issues[1].property, "name");
    assert_eq!(issues[2].entity, ExternalEntityRef::Edge(edge));
    assert!(!graph.has_uncommitted_changes());
    graph
        .set_node_attrs(HashMap::from([
            (
                "name".into(),
                vec![(node, AttrValue::Text("x".repeat(1000)))],
            ),
            ("age".into(), vec![(node, AttrValue::Int(42))]),
        ]))
        .unwrap();
    graph
        .set_edge_attr(edge, "weight".into(), AttrValue::Float(1.0))
        .unwrap();
    assert!(graph.validate_schema(&schema).unwrap().is_empty());
    for bad in [
        r#"{"node_labels":{"":{"x":{"type":"int"}}}}"#,
        r#"{"unknown":{}}"#,
        r#"{"node_labels":{"Entity":{"x":{"type":"unknown"}}}}"#,
    ] {
        assert!(GraphSchema::from_json(bad).is_err());
    }
}

#[test]
fn kinds_and_schema_applicability_survive_typed_export() {
    let mut graph = Graph::new();
    let node = graph.add_node();
    graph.set_node_labels(node, vec!["Entity".into()]).unwrap();
    let edge = graph.add_edge(node, node).unwrap();
    graph
        .set_edge_relationship_type(edge, Some("IS".into()))
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.json");
    graph.save_to_path(&path).unwrap();
    let loaded = Graph::load_from_path(&path).unwrap();
    assert_eq!(loaded.nodes_with_label("Entity").unwrap(), vec![node]);
    assert_eq!(
        loaded.edges_with_relationship_type("IS").unwrap(),
        vec![edge]
    );
    let schema =
        GraphSchema::from_json(r#"{"node_labels":{"Entity":{"required":{"type":"int"}}}}"#)
            .unwrap();
    assert_eq!(loaded.validate_schema(&schema).unwrap().len(), 1);
}
