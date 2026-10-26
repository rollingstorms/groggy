use groggy::state::change_tracker::ChangeSet;
use groggy::{AttrValue as V, CommitDiffPlan, Graph, HistoryForest, RecordedReadPlan};
fn assert_thread_safe<T: Send + Sync>() {}

#[test]
fn captured_diff_survives_mutation_pruning_purge_and_owner_destruction() {
    assert_thread_safe::<CommitDiffPlan>();
    assert_thread_safe::<RecordedReadPlan>();
    let mut graph = Graph::new();
    graph.add_node().unwrap();
    let root = graph.current_state_id();
    graph.create_branch("discard".into()).unwrap();
    graph.checkout_branch("discard".into()).unwrap();
    let node = graph.add_node().unwrap();
    graph
        .set_node_attr(node, "value".into(), V::Text("retained by delta".into()))
        .unwrap();
    let edge = graph.add_edge(0, node).unwrap();
    let future = graph.current_state_id();
    let plan = graph.prepare_commit_diff(root, future).unwrap();
    graph.checkout_branch("main".into()).unwrap();
    graph.prune_branch("discard").unwrap();
    graph.purge_unreferenced().unwrap();
    assert!(!graph.is_node_allocated(node));
    assert!(graph.prepare_commit_diff(root, future).is_err());
    graph.add_node().unwrap();
    drop(graph);
    let repeated = plan.clone();
    let diff = std::thread::spawn(move || plan.execute().unwrap())
        .join()
        .unwrap();
    assert_eq!((diff.from_commit, diff.to_commit), (root, future));
    assert_eq!(diff.nodes_added, vec![node]);
    assert_eq!(diff.edges_added, vec![(edge, 0, node)]);
    assert!(diff.attr_changes.iter().any(
        |v| v.attr_name == "value" && v.new_value == Some(V::Text("retained by delta".into()))
    ));
    assert_eq!(repeated.execute().unwrap().nodes_added, diff.nodes_added);
}

#[test]
fn recorded_plan_zero_unknown_and_resolved_merge_follow_first_parent_semantics() {
    let mut history = HistoryForest::new();
    let root = history
        .create_commit(
            ChangeSet {
                nodes_added: vec![0],
                ..ChangeSet::new()
            },
            "root".into(),
            "test".into(),
            None,
        )
        .unwrap();
    let left = history
        .create_commit(
            ChangeSet {
                node_attr_changes: vec![(0, "left".into(), None, Some(V::Int(1)))],
                ..ChangeSet::new()
            },
            "left".into(),
            "test".into(),
            Some(root),
        )
        .unwrap();
    let right_changes = || ChangeSet {
        node_attr_changes: vec![(0, "right".into(), None, Some(V::Int(2)))],
        ..ChangeSet::new()
    };
    let right = history
        .create_commit(right_changes(), "right".into(), "test".into(), Some(root))
        .unwrap();
    let merge = history
        .create_merge_commit(
            right_changes(),
            "resolved".into(),
            "test".into(),
            vec![left, right],
        )
        .unwrap();
    let read = history.prepare_recorded_read(merge).unwrap();
    let diff = history.prepare_commit_diff(left, merge).unwrap();
    assert!(history.prepare_recorded_read(999).is_err());
    assert!(history.prepare_commit_diff(0, 999).is_err());
    assert!(history.prepare_commit_diff(999, 0).is_err());
    assert!(history
        .prepare_recorded_read(0)
        .unwrap()
        .execute()
        .unwrap()
        .active_nodes
        .is_empty());
    drop(history);
    let state = std::thread::spawn(move || read.execute().unwrap())
        .join()
        .unwrap();
    assert_eq!(state.node_attributes[&0]["left"], V::Int(1));
    assert_eq!(state.node_attributes[&0]["right"], V::Int(2));
    let changes = diff.execute().unwrap();
    assert_eq!(changes.attr_changes.len(), 1);
    assert_eq!(changes.attr_changes[0].attr_name, "right");
    assert!(changes.edge_endpoint_changes.is_empty());
}
