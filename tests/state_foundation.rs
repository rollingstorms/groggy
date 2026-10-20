use groggy::{AttrValue, Graph};

#[test]
fn live_edge_reads_respect_membership_and_direction() {
    let mut graph = Graph::new_directed();
    let source = graph.add_node();
    let target = graph.add_node();
    let edge = graph.add_edge(source, target).unwrap();

    assert!(graph.has_edge_between(source, target).unwrap());
    assert!(!graph.has_edge_between(target, source).unwrap());
    assert_eq!(graph.edge_endpoints(edge).unwrap(), (source, target));

    graph.remove_edge(edge).unwrap();
    assert!(!graph.has_edge_between(source, target).unwrap());
    assert!(graph.edge_endpoints(edge).is_err());
}

#[test]
fn invalid_bulk_edges_report_every_position_without_mutating() {
    let mut graph = Graph::new();
    let node = graph.add_node();
    let error = graph
        .add_edges(&[(node, node), (node, 99), (100, node)])
        .unwrap_err()
        .to_string();
    assert!(error.contains("1: (0, 99)"), "{error}");
    assert!(error.contains("2: (100, 0)"), "{error}");
    assert!(graph.edge_ids().is_empty());
    assert_eq!(graph.add_edge(node, node).unwrap(), 0);
}

#[test]
fn historical_snapshots_keep_real_typed_property_values_and_removals() {
    let mut graph = Graph::new();
    let a = graph.add_node();
    let b = graph.add_node();
    let edge = graph.add_edge(a, b).unwrap();
    graph
        .set_node_attr(a, "score".into(), AttrValue::Int(1))
        .unwrap();
    graph
        .set_node_attr(a, "score".into(), AttrValue::FloatVec(vec![1.0, 2.0]))
        .unwrap();
    graph
        .set_edge_attr(edge, "label".into(), AttrValue::Text("first".into()))
        .unwrap();
    let first = graph.commit("first".into(), "test".into()).unwrap();

    graph
        .set_node_attr(a, "score".into(), AttrValue::Text("later".into()))
        .unwrap();
    graph
        .set_edge_attr(edge, "label".into(), AttrValue::Text("second".into()))
        .unwrap();
    graph.remove_edge(edge).unwrap();
    let second = graph.commit("second".into(), "test".into()).unwrap();

    let before = graph.snapshot_at_commit(first).unwrap();
    assert_eq!(
        before.node_attr(a, &"score".into()),
        Some(AttrValue::FloatVec(vec![1.0, 2.0]))
    );
    assert_eq!(
        before.node_attr(a, &"entity_type".into()),
        Some(AttrValue::Text("base".into()))
    );
    assert_eq!(
        before.edge_attr(edge, &"label".into()),
        Some(AttrValue::Text("first".into()))
    );
    assert!(before.edge_exists(edge));

    let after = graph.snapshot_at_commit(second).unwrap();
    assert_eq!(
        after.node_attr(a, &"score".into()),
        Some(AttrValue::Text("later".into()))
    );
    assert!(!after.edge_exists(edge));
    assert!(!graph.has_edge(edge));
}

#[test]
fn snapshots_preserve_graph_direction() {
    for directed in [false, true] {
        let mut graph = if directed {
            Graph::new_directed()
        } else {
            Graph::new_undirected()
        };
        let a = graph.add_node();
        let b = graph.add_node();
        graph.add_edge(a, b).unwrap();
        let state = graph.commit("topology".into(), "test".into()).unwrap();
        let snapshot = graph.snapshot_at_commit(state).unwrap();
        assert_eq!(snapshot.neighbors(a).unwrap(), &vec![b]);
        let expected_reverse = if directed { vec![] } else { vec![a] };
        assert_eq!(snapshot.neighbors(b).unwrap(), &expected_reverse);
        let restored = snapshot.as_subgraph().unwrap();
        assert_eq!(restored.graph().borrow().is_directed(), directed);
    }
}

#[test]
fn switching_to_older_branch_does_not_reuse_historical_ids() {
    let mut graph = Graph::new();
    let root = graph.add_node();
    graph.commit("root".into(), "test".into()).unwrap();
    graph.create_branch("feature".into()).unwrap();
    graph.checkout_branch("feature".into()).unwrap();
    let feature_node = graph.add_node();
    let feature_edge = graph.add_edge(root, feature_node).unwrap();
    graph.commit("feature".into(), "test".into()).unwrap();

    graph.checkout_branch("main".into()).unwrap();
    let main_node = graph.add_node();
    let main_edge = graph.add_edge(root, main_node).unwrap();
    assert!(main_node > feature_node);
    assert!(main_edge > feature_edge);
}

#[test]
fn reset_hard_restores_published_state_and_preserves_allocated_ids() {
    let mut graph = Graph::new();
    let root = graph.add_node();
    graph.commit("root".into(), "test".into()).unwrap();
    let discarded = graph.add_node();
    let edge = graph.add_edge(root, discarded).unwrap();
    graph
        .set_node_attr(root, "temporary".into(), AttrValue::Int(42))
        .unwrap();
    graph.reset_hard().unwrap();

    assert!(graph.has_node(root));
    assert!(!graph.has_node(discarded));
    assert!(!graph.has_edge(edge));
    assert_eq!(
        graph.get_node_attr(root, &"temporary".into()).unwrap(),
        None
    );
    assert_eq!(graph.add_node(), discarded + 1);
}
