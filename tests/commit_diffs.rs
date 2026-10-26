use groggy::{AttrValue, Graph};

#[test]
fn diff_commits_compares_typed_state_in_both_directions() {
    let mut graph = Graph::new_manual();
    let alice = graph.add_node().unwrap();
    graph
        .set_node_attr(alice, "score".into(), AttrValue::Int(1))
        .unwrap();
    let first = graph.commit("first".into(), "test".into()).unwrap();

    let bob = graph.add_node().unwrap();
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
    let mut graph = Graph::new_manual();
    graph.add_node().unwrap();
    let state = graph.commit("first".into(), "test".into()).unwrap();
    assert_eq!(graph.diff_commits(0, state).unwrap().nodes_added, vec![0]);
    assert!(graph.diff_commits(state, 999).is_err());
}

#[test]
fn selected_snapshot_diff_reports_ordered_endpoint_changes_without_fake_additions() {
    use groggy::state::state::GraphSnapshot;
    use groggy::types::GraphType;
    use groggy::EdgeEndpointChange;
    for kind in [GraphType::Directed, GraphType::Undirected] {
        let mut before = GraphSnapshot::empty_with_type(1, kind);
        before.active_nodes = vec![0, 1, 2];
        before
            .edges
            .extend([(7, (0, 1)), (2, (0, 1)), (9, (2, 2)), (11, (1, 2))]);
        let mut after = before.clone();
        after.state_id = 2;
        after.edges.insert(7, (2, 1));
        after.edges.insert(2, (1, 0));
        after.edges.remove(&11);
        after.edges.insert(20, (0, 0));
        let expected = vec![
            EdgeEndpointChange {
                edge_id: 2,
                old_endpoints: (0, 1),
                new_endpoints: (1, 0),
            },
            EdgeEndpointChange {
                edge_id: 7,
                old_endpoints: (0, 1),
                new_endpoints: (2, 1),
            },
        ];
        let diff = before.diff_with(&after);
        assert_eq!(diff.edge_endpoint_changes, expected);
        assert_eq!(diff.edges_added, vec![(20, 0, 0)]);
        assert_eq!(diff.edges_removed, vec![11]);
        assert_eq!(diff.summary().edges_changed, 4);
        assert!(diff.attribute_changes.is_empty());
        let reverse = after.diff_with(&before);
        for (forward, backward) in diff
            .edge_endpoint_changes
            .iter()
            .zip(&reverse.edge_endpoint_changes)
        {
            assert_eq!(forward.edge_id, backward.edge_id);
            assert_eq!(forward.old_endpoints, backward.new_endpoints);
            assert_eq!(forward.new_endpoints, backward.old_endpoints);
        }
        assert!(before.diff_with(&before).is_empty());
        let mut endpoint_only = before.clone();
        endpoint_only.edges.insert(9, (1, 1));
        let diff = before.diff_with(&endpoint_only);
        assert!(!diff.is_empty());
        assert_eq!(diff.summary().edges_changed, 1);
        assert!(diff.edges_added.is_empty() && diff.edges_removed.is_empty());
    }
}
