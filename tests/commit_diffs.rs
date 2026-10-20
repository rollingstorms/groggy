use groggy::{AttrValue, Graph};

#[test]
fn diff_commits_compares_typed_state_in_both_directions() {
    let mut graph = Graph::new();
    let alice = graph.add_node();
    graph
        .set_node_attr(alice, "score".into(), AttrValue::Int(1))
        .unwrap();
    let first = graph.commit("first".into(), "test".into()).unwrap();

    let bob = graph.add_node();
    let edge = graph.add_edge(alice, bob).unwrap();
    graph
        .set_node_attr(alice, "score".into(), AttrValue::FloatVec(vec![1.0, 2.0]))
        .unwrap();
    graph
        .set_edge_attr(edge, "weight".into(), AttrValue::Float(1.5))
        .unwrap();
    let second = graph.commit("second".into(), "test".into()).unwrap();

    let forward = graph.diff_commits(first, second).unwrap();
    assert_eq!(forward.nodes_added, vec![bob]);
    assert!(forward.nodes_removed.is_empty());
    assert_eq!(forward.edges_added, vec![(edge, alice, bob)]);
    assert!(forward.attr_changes.iter().any(|change| {
        change.entity_id == alice as u64
            && change.attr_name == "score"
            && change.old_value == Some(AttrValue::Int(1))
            && change.new_value == Some(AttrValue::FloatVec(vec![1.0, 2.0]))
    }));
    assert!(forward.attr_changes.iter().any(|change| {
        change.entity_id == edge as u64
            && change.attr_name == "weight"
            && change.old_value.is_none()
            && change.new_value == Some(AttrValue::Float(1.5))
    }));

    let reverse = graph.diff_commits(second, first).unwrap();
    assert_eq!(reverse.nodes_removed, vec![bob]);
    assert_eq!(reverse.edges_removed, vec![edge]);
    assert!(reverse.attr_changes.iter().any(|change| {
        change.attr_name == "score"
            && change.old_value == Some(AttrValue::FloatVec(vec![1.0, 2.0]))
            && change.new_value == Some(AttrValue::Int(1))
    }));
    assert!(graph
        .diff_commits(first, first)
        .unwrap()
        .nodes_added
        .is_empty());
}

#[test]
fn diff_rejects_unknown_state_instead_of_treating_it_as_empty() {
    let mut graph = Graph::new();
    graph.add_node();
    let state = graph.commit("first".into(), "test".into()).unwrap();
    assert_eq!(graph.diff_commits(0, state).unwrap().nodes_added, vec![0]);
    assert!(graph.diff_commits(state, 999).is_err());
}
