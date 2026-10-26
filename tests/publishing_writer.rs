use groggy::{AttrValue, Graph, GraphSchema};
use std::collections::HashMap;
#[test]
fn checked_direct_and_bulk_writes_publish_once_per_call() {
    let mut graph = Graph::new_manual();
    let mut writer = graph.writer("test");
    let nodes = writer
        .add_nodes_with_attrs(vec![
            HashMap::from([("name".into(), AttrValue::Text("Ada".into()))]),
            HashMap::new(),
        ])
        .unwrap();
    assert_eq!(writer.graph().commit_history().len(), 1);
    let initial = writer.state_id();
    let snapshot = writer.graph().snapshot_at_commit(initial).unwrap();
    let edge = writer
        .add_edge(
            nodes[0],
            nodes[1],
            HashMap::from([("weight".into(), AttrValue::Float(0.5))]),
        )
        .unwrap();
    assert_eq!(writer.graph().commit_history().len(), 2);
    writer
        .set_node_attrs(HashMap::from([(
            "name".into(),
            vec![
                (nodes[0], AttrValue::Text("Grace".into())),
                (nodes[1], AttrValue::Text("Lin".into())),
            ],
        )]))
        .unwrap();
    assert_eq!(writer.graph().commit_history().len(), 3);
    assert!(!writer.graph().has_uncommitted_changes());
    assert_eq!(
        snapshot
            .node_attr(nodes[0], &"name".into())
            .unwrap()
            .get_text()
            .as_deref(),
        Some("Ada")
    );
    writer.remove_nodes(vec![nodes[0]]).unwrap();
    assert!(!writer.graph().has_edge(edge));
    assert_eq!(writer.graph().commit_history().len(), 4);
}
#[test]
fn schema_policy_applies_to_every_session_write_and_rejections_preserve_head() {
    let mut graph = Graph::new_manual();
    let schema =
        GraphSchema::from_json(r#"{"node_labels":{"Entity":{"name":{"type":"text"}}}}"#).unwrap();
    let mut writer = graph.writer("test").with_schema(&schema);
    let attrs = HashMap::from([("labels".into(), AttrValue::TextVec(vec!["Entity".into()]))]);
    assert!(writer.add_node(attrs.clone()).is_err());
    assert_eq!(writer.state_id(), 0);
    let mut attrs = attrs;
    attrs.insert("name".into(), AttrValue::Text("Ada".into()));
    assert_eq!(writer.add_node(attrs).unwrap(), 0);
    let state = writer.state_id();
    assert!(writer
        .set_node_attr(0, "name".into(), AttrValue::Int(3))
        .is_err());
    assert_eq!(writer.state_id(), state);
    assert!(!writer.graph().has_uncommitted_changes());
}
