use groggy::{AttrValue, BatchEdgeRef as E, BatchNodeRef as N, Graph, GraphBatch, GraphSchema};
use std::collections::HashMap;

fn schema() -> GraphSchema {
    GraphSchema::from_json(r#"{
        "node_labels":{"Entity":{"name":{"type":"text"},"entity_type":{"type":"text"},"labels":{"type":"text_vec"}}},
        "relationship_types":{"KNOWS":{"weight":{"type":"float"},"note":{"type":"text","required":false,"nullable":true}}}
    }"#).unwrap()
}
fn people_batch() -> GraphBatch {
    GraphBatch {
        add_nodes: 2,
        add_edges: vec![(N::Added(0), N::Added(1))],
        node_attrs: HashMap::from([
            (
                "labels".into(),
                vec![(N::Added(0), AttrValue::TextVec(vec!["Entity".into()]))],
            ),
            (
                "name".into(),
                vec![(N::Added(0), AttrValue::Text("Ada".into()))],
            ),
        ]),
        edge_attrs: HashMap::from([
            (
                "relationship_type".into(),
                vec![(E::Added(0), AttrValue::Text("KNOWS".into()))],
            ),
            ("weight".into(), vec![(E::Added(0), AttrValue::Float(0.5))]),
        ]),
        ..Default::default()
    }
}

#[test]
fn schema_batch_publishes_once_and_preserves_prior_snapshot() {
    let mut graph = Graph::new_directed_manual();
    let first = graph
        .apply_batch_with_schema(people_batch(), 0, "people".into(), "test".into(), &schema())
        .unwrap();
    assert_eq!(first.added_nodes, vec![0, 1]);
    assert_eq!(first.added_edges, vec![0]);
    assert_eq!(graph.commit_history().len(), 1);
    assert!(!graph.has_uncommitted_changes());
    assert!(graph.validate_schema(&schema()).unwrap().is_empty());
    let prior = graph.snapshot_at_commit(first.state_id).unwrap();
    let update = GraphBatch {
        node_attrs: HashMap::from([(
            "name".into(),
            vec![(N::Existing(0), AttrValue::Text("Grace".into()))],
        )]),
        ..Default::default()
    };
    graph
        .apply_batch_with_schema(
            update,
            first.state_id,
            "rename".into(),
            "test".into(),
            &schema(),
        )
        .unwrap();
    assert_eq!(graph.commit_history().len(), 2);
    assert_eq!(
        prior
            .node_attr(0, &"name".into())
            .unwrap()
            .get_text()
            .as_deref(),
        Some("Ada")
    );
}

#[test]
fn all_schema_findings_reject_before_ids_properties_or_history_change() {
    let mut graph = Graph::new_manual();
    let mut batch = people_batch();
    batch.node_attrs.remove("name");
    batch
        .edge_attrs
        .insert("weight".into(), vec![(E::Added(0), AttrValue::Int(2))]);
    batch
        .edge_attrs
        .insert("note".into(), vec![(E::Added(0), AttrValue::Bool(false))]);
    let message = graph
        .apply_batch_with_schema(batch, 0, "invalid".into(), "test".into(), &schema())
        .unwrap_err()
        .to_string();
    assert!(message.contains("node 0 [Entity].name"));
    assert!(message.contains("edge 0 [KNOWS].note"));
    assert!(message.contains("edge 0 [KNOWS].weight"));
    assert_eq!(graph.current_state_id(), 0);
    assert_eq!(graph.commit_history().len(), 0);
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(graph.add_node().unwrap(), 0);
    assert_eq!(graph.add_edge(0, 0).unwrap(), 0);
}

#[test]
fn label_and_required_property_changes_are_validated_together() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    graph
        .set_node_attr(node, "name".into(), AttrValue::Int(42))
        .unwrap();
    let first = graph.commit("untyped".into(), "test".into()).unwrap();
    let update = GraphBatch {
        node_attrs: HashMap::from([
            (
                "labels".into(),
                vec![(N::Existing(node), AttrValue::TextVec(vec!["Entity".into()]))],
            ),
            (
                "name".into(),
                vec![(N::Existing(node), AttrValue::Text("Ada".into()))],
            ),
        ]),
        ..Default::default()
    };
    graph
        .apply_batch_with_schema(
            update,
            first,
            "type and label".into(),
            "test".into(),
            &schema(),
        )
        .unwrap();
    assert!(graph.validate_schema(&schema()).unwrap().is_empty());
}

#[test]
fn unchanged_invalid_entities_are_checked_but_removed_entities_are_excluded() {
    let mut graph = Graph::new_manual();
    let nodes = graph.add_nodes(2).unwrap();
    graph
        .set_node_labels(nodes[0], vec!["Entity".into()])
        .unwrap();
    let edge = graph.add_edge(nodes[0], nodes[1]).unwrap();
    graph
        .set_edge_relationship_type(edge, Some("KNOWS".into()))
        .unwrap();
    let state = graph
        .commit("invalid legacy fixture".into(), "test".into())
        .unwrap();
    let unrelated = GraphBatch {
        add_nodes: 1,
        ..Default::default()
    };
    assert!(graph
        .apply_batch_with_schema(
            unrelated,
            state,
            "unrelated".into(),
            "test".into(),
            &schema()
        )
        .is_err());
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(graph.current_state_id(), state);
    // Removing the node also removes the invalid incident edge in the proposed state.
    let remove = GraphBatch {
        remove_nodes: vec![N::Existing(nodes[0])],
        ..Default::default()
    };
    graph
        .apply_batch_with_schema(remove, state, "remove".into(), "test".into(), &schema())
        .unwrap();
    assert!(!graph.has_edge(edge));
    assert!(graph.validate_schema(&schema()).unwrap().is_empty());
}

#[test]
fn clearing_selectors_and_removing_edges_use_final_state_rules() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    graph.set_node_labels(node, vec!["Entity".into()]).unwrap();
    let edge = graph.add_edge(node, node).unwrap();
    graph
        .set_edge_relationship_type(edge, Some("KNOWS".into()))
        .unwrap();
    let state = graph
        .commit("invalid selectors".into(), "test".into())
        .unwrap();
    let clear = GraphBatch {
        node_attrs: HashMap::from([(
            "labels".into(),
            vec![(N::Existing(node), AttrValue::FloatVec(vec![]))],
        )]),
        remove_edges: vec![E::Existing(edge)],
        ..Default::default()
    };
    graph
        .apply_batch_with_schema(
            clear,
            state,
            "clear selectors".into(),
            "test".into(),
            &schema(),
        )
        .unwrap();
    assert_eq!(graph.node_labels(node).unwrap(), Vec::<String>::new());
}

#[test]
fn failed_schema_batch_respects_retained_allocation_high_water() {
    let mut graph = Graph::new_manual();
    let removed = graph.add_node().unwrap();
    graph.remove_node(removed).unwrap();
    let head = graph.commit("removed".into(), "test".into()).unwrap();
    let mut bad = people_batch();
    bad.node_attrs.remove("name");
    let error = graph
        .apply_batch_with_schema(bad, head, "bad".into(), "test".into(), &schema())
        .unwrap_err()
        .to_string();
    assert!(error.contains("node 1 [Entity].name"));
    assert_eq!(graph.add_node().unwrap(), 1);
}

#[test]
fn stale_and_dirty_guards_also_apply_to_schema_batches() {
    let mut graph = Graph::new_manual();
    graph
        .apply_batch_with_schema(
            people_batch(),
            0,
            "initial".into(),
            "test".into(),
            &schema(),
        )
        .unwrap();
    assert!(graph
        .apply_batch_with_schema(people_batch(), 0, "stale".into(), "test".into(), &schema())
        .unwrap_err()
        .to_string()
        .contains("Stale"));
    let head = graph.current_state_id();
    graph
        .set_node_attr(0, "free".into(), AttrValue::Int(1))
        .unwrap();
    assert!(graph
        .apply_batch_with_schema(
            people_batch(),
            head,
            "dirty".into(),
            "test".into(),
            &schema()
        )
        .unwrap_err()
        .to_string()
        .contains("uncommitted"));
    assert_eq!(graph.current_state_id(), head);
    assert_eq!(
        graph.get_node_attr(0, &"free".into()).unwrap(),
        Some(AttrValue::Int(1))
    );
}
