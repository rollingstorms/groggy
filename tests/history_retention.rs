use groggy::{AttrValue as V, BatchNodeRef as N, Graph, GraphBatch};
use std::collections::HashMap;
fn initial(graph: &mut Graph) -> u64 {
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                node_attrs: HashMap::from([("v".into(), vec![(N::Added(0), V::Int(0))])]),
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "first".into(),
            "test".into(),
        )
        .unwrap()
        .state_id
}
#[test]
fn explicit_pruning_preserves_shared_ancestry_open_reads_and_lifetime_reservations() {
    let mut graph = Graph::new_publishing();
    let root = initial(&mut graph);
    graph
        .checkpoint("root label".into(), "test".into())
        .unwrap();
    graph.create_branch("discard".into()).unwrap();
    graph.checkout_branch("discard".into()).unwrap();
    graph
        .writer("test")
        .with_effective_at(Some(20))
        .set_node_attr(0, "v".into(), V::Int(1))
        .unwrap();
    let shared = graph.current_state_id();
    graph.create_branch("shared".into()).unwrap();
    graph
        .writer("test")
        .with_effective_at(Some(30))
        .set_node_attr(0, "v".into(), V::Int(2))
        .unwrap();
    let only = graph.current_state_id();
    graph
        .checkpoint("do not pin".into(), "test".into())
        .unwrap();
    let pin = graph.read_view().unwrap();
    let plan = graph.prepare_read_at(30, None).unwrap();
    let recorded = graph.snapshot_at_commit(only).unwrap();
    let index = graph.build_temporal_index().unwrap();
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                node_external_ids: vec![(N::Added(0), "discarded-alias".into())],
                effective_at: Some(40),
                ..Default::default()
            },
            only,
            "extra".into(),
            "test".into(),
        )
        .unwrap();
    graph.checkout_branch("main".into()).unwrap();
    assert_eq!(graph.prune_branch("discard").unwrap(), 2);
    assert_eq!(graph.current_state_id(), root);
    assert_eq!(graph.history_records().len(), 2);
    assert!(graph.snapshot_at_commit(only).is_err());
    assert_eq!(
        graph
            .snapshot_at_commit(shared)
            .unwrap()
            .node_attr(0, &"v".into()),
        Some(V::Int(1))
    );
    assert_eq!(pin.node_attr(0, &"v".into()).unwrap(), Some(V::Int(2)));
    assert_eq!(
        plan.read_view().unwrap().node_attr(0, &"v".into()).unwrap(),
        Some(V::Int(2))
    );
    assert_eq!(recorded.node_attr(0, &"v".into()), Some(V::Int(2)));
    assert_eq!(
        index.node_attr_at_commit(0, &"v".into(), only),
        Some(V::Int(2))
    );
    assert_eq!(graph.checkpoints().len(), 1);
    assert_eq!(
        graph.allocated_entity_for_external_id("discarded-alias"),
        Some(groggy::ExternalEntityRef::Node(1))
    );
    assert_eq!(graph.try_add_node().unwrap(), 2);
    assert!(graph.current_state_id() > only + 1);
    assert_eq!(graph.gc_history().unwrap(), 0);
}
#[test]
fn collector_removes_unreferenced_owned_future_states_and_keeps_capture_independent() {
    let mut graph = Graph::new_publishing();
    let root = initial(&mut graph);
    graph
        .writer("test")
        .set_node_attr(0, "v".into(), V::Int(9))
        .unwrap();
    let future = graph.current_state_id();
    let capture = graph.owned_snapshot_at_commit(root).unwrap();
    let mut copy = capture.to_graph().unwrap();
    assert_eq!(copy.gc_history().unwrap(), 1);
    assert!(copy.snapshot_at_commit(future).is_err());
    assert_eq!(copy.history_records().len(), 1);
    assert_eq!(
        capture
            .to_graph()
            .unwrap()
            .snapshot_at_commit(future)
            .unwrap()
            .node_attr(0, &"v".into()),
        Some(V::Int(9))
    );
    assert_eq!(
        graph
            .snapshot_at_commit(future)
            .unwrap()
            .node_attr(0, &"v".into()),
        Some(V::Int(9))
    );
}
#[test]
fn invalid_current_main_unknown_and_dirty_prunes_leave_every_reference_intact() {
    let mut graph = Graph::new_publishing();
    initial(&mut graph);
    graph.create_branch("other".into()).unwrap();
    for name in ["main", "unknown"] {
        assert!(graph.prune_branch(name).is_err());
    }
    graph.checkout_branch("other".into()).unwrap();
    assert!(graph.prune_branch("other").is_err());
    graph.set_automatic_publication(false).unwrap();
    graph.set_node_attr(0, "pending".into(), V::Null).unwrap();
    assert!(graph.gc_history().is_err());
    assert!(graph.prune_branch("main").is_err());
    assert_eq!(graph.list_branches().len(), 2);
    assert!(graph.has_uncommitted_changes());
}
#[test]
fn raw_history_collection_keeps_requested_ancestry_and_branch_roots_and_cleans_children() {
    use groggy::state::{change_tracker::ChangeSet, history::HistoryForest};
    let mut history = HistoryForest::new();
    assert!(history.find_reachable_commits().is_empty());
    let root = history
        .create_commit(ChangeSet::new(), "root".into(), "test".into(), None)
        .unwrap();
    let kept = history
        .create_commit(ChangeSet::new(), "kept".into(), "test".into(), Some(root))
        .unwrap();
    let gone = history
        .create_commit(ChangeSet::new(), "gone".into(), "test".into(), Some(root))
        .unwrap();
    history.create_branch("kept".into(), kept).unwrap();
    assert_eq!(history.garbage_collect(&Default::default()), 1);
    assert!(history.get_commit(gone).is_err());
    assert!(history.get_commit(root).is_ok());
    assert_eq!(history.get_children(root), vec![kept]);
    let extra = history
        .create_commit(ChangeSet::new(), "extra".into(), "test".into(), Some(kept))
        .unwrap();
    assert!(extra > gone);
    assert_eq!(history.garbage_collect(&[extra].into_iter().collect()), 0);
}

#[test]
fn collection_keeps_both_merge_parent_lineages_and_drops_only_unreferenced_descendants() {
    use groggy::state::{change_tracker::ChangeSet, history::HistoryForest};
    let mut history = HistoryForest::new();
    let root = history
        .create_commit(ChangeSet::new(), "root".into(), "test".into(), None)
        .unwrap();
    let left = history
        .create_commit(ChangeSet::new(), "left".into(), "test".into(), Some(root))
        .unwrap();
    let right = history
        .create_commit(ChangeSet::new(), "right".into(), "test".into(), Some(root))
        .unwrap();
    let merged = history
        .create_merge_commit_with_publication(
            ChangeSet::new(),
            "merged".into(),
            "test".into(),
            vec![left, right],
            None,
        )
        .unwrap();
    history.update_branch_head(&"main".into(), merged).unwrap();
    let orphan = history
        .create_commit(
            ChangeSet::new(),
            "orphan".into(),
            "test".into(),
            Some(right),
        )
        .unwrap();
    assert_eq!(history.garbage_collect(&Default::default()), 1);
    assert_eq!(history.find_reachable_commits().len(), 4);
    assert!(history.get_commit(left).is_ok() && history.get_commit(right).is_ok());
    assert!(history.get_commit(orphan).is_err());
    assert_eq!(history.get_children(right), vec![merged]);
}
