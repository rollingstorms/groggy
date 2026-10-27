use groggy::{
    AttrValue as V, BatchNodeRef as N, ChangeKey as K, Graph, GraphBatch, GraphMergeOptions,
    StateRestoreMode as M,
};
fn fixture() -> (Graph, u64, u64) {
    let mut g = Graph::new();
    g.apply_batch(
        GraphBatch {
            add_nodes: 3,
            add_edges: vec![
                (N::Added(0), N::Added(1)),
                (N::Added(0), N::Added(1)),
                (N::Added(1), N::Added(1)),
            ],
            node_attrs: [(
                "value".into(),
                vec![(N::Added(0), V::Float64(-0.0)), (N::Added(1), V::Null)],
            )]
            .into(),
            effective_at: Some(10),
            ..Default::default()
        },
        0,
        "root".into(),
        "test".into(),
    )
    .unwrap();
    let old = g.current_state_id();
    g.apply_batch(
        GraphBatch {
            node_attrs: [(
                "value".into(),
                vec![(N::Existing(0), V::Text("changed".into()))],
            )]
            .into(),
            edge_endpoints: vec![(0, N::Existing(2), N::Existing(1))],
            effective_at: Some(20),
            ..Default::default()
        },
        old,
        "edit".into(),
        "test".into(),
    )
    .unwrap();
    g.writer("test")
        .with_effective_at(Some(30))
        .remove_nodes(vec![1])
        .unwrap();
    let current = g.current_state_id();
    (g, old, current)
}
fn zero(g: &Graph) {
    let Some(V::Float64(x)) = g.get_node_attr(0, &"value".into()).unwrap() else {
        panic!("not f64")
    };
    assert_eq!(x.to_bits(), (-0.0_f64).to_bits());
}
#[test]
fn tracked_restore_publishes_one_exact_single_parent_delta_and_preserves_history_views_and_clocks()
{
    let (mut g, old, current) = fixture();
    g.create_branch("future".into()).unwrap();
    let pin = g.read_view().unwrap();
    let versions = g.history_records().len();
    let original = g.publication_metadata(old).unwrap();
    let plan = g.prepare_state_restore(old, current, M::Tracked).unwrap();
    let outcome = std::thread::spawn(move || plan.execute().unwrap())
        .join()
        .unwrap();
    let changes = outcome.changes().unwrap();
    assert_eq!(changes.nodes_added, vec![1]);
    assert_eq!(changes.edges_added.len(), 3);
    assert!(changes.nodes_removed.is_empty());
    g.compact_pool_with_reclamation(true).unwrap();
    let id = g
        .finish_state_restore_with_options(
            outcome,
            GraphMergeOptions {
                effective_at: Some(50),
                change_times: [(K::NodeAdded(1), 40)].into(),
            },
            "undo edits".into(),
            "owner".into(),
            None,
        )
        .unwrap();
    assert!(id > current);
    assert_eq!(g.history_records().len(), versions + 1);
    let record = g
        .history_records()
        .into_iter()
        .find(|c| c.id == id)
        .unwrap();
    assert_eq!(record.parents, vec![current]);
    assert_eq!(record.author, "owner");
    assert_eq!(g.publication_metadata(old).unwrap(), original);
    let metadata = g.publication_metadata(id).unwrap().unwrap();
    assert_eq!(
        metadata.change_times[&K::NodeAttribute(1, "value".into())],
        40
    );
    assert!(!pin.contains_node(1));
    assert!(g.has_node(1));
    assert_eq!(g.edge_endpoints(0).unwrap(), (0, 1));
    assert_eq!(g.edge_endpoints(1).unwrap(), (0, 1));
    assert_eq!(g.edge_endpoints(2).unwrap(), (1, 1));
    zero(&g);
    assert!(!g.read_view_at(39, None).unwrap().contains_node(1));
    assert!(g.read_view_at(40, None).unwrap().contains_node(1));
    let cutoff = g
        .publication_metadata(current)
        .unwrap()
        .unwrap()
        .recorded_at;
    assert!(!g.read_view_at(100, Some(cutoff)).unwrap().contains_node(1));
    let restored = g.prepare_archive().unwrap().restore().unwrap();
    assert_eq!(restored.current_state_id(), id);
    zero(&restored);
    g.check_integrity().unwrap();
    assert_eq!(g.add_node().unwrap(), 3);
    g.checkout_branch("future".into()).unwrap();
    assert_eq!(g.current_state_id(), current);
    assert!(!g.has_node(1));
}
#[test]
fn tracked_restore_rejects_bad_times_stale_foreign_and_collected_inputs_without_publication() {
    let (mut g, old, current) = fixture();
    let versions = g.history_records().len();
    for options in [
        GraphMergeOptions {
            effective_at: Some(5),
            ..Default::default()
        },
        GraphMergeOptions {
            change_times: [(K::NodeAdded(999), 50)].into(),
            ..Default::default()
        },
    ] {
        let outcome = g
            .prepare_state_restore(old, current, M::Tracked)
            .unwrap()
            .execute()
            .unwrap();
        assert!(g
            .finish_state_restore_with_options(outcome, options, "bad".into(), "test".into(), None)
            .is_err());
        assert_eq!(g.current_state_id(), current);
        assert_eq!(g.history_records().len(), versions);
        assert!(!g.has_node(1));
    }
    let outcome = g
        .prepare_state_restore(old, current, M::Tracked)
        .unwrap()
        .execute()
        .unwrap();
    let mut foreign = g.prepare_archive().unwrap().restore().unwrap();
    assert!(foreign.finish_state_restore(outcome).is_err());
    let outcome = g
        .prepare_state_restore(old, current, M::Tracked)
        .unwrap()
        .execute()
        .unwrap();
    g.create_branch("same".into()).unwrap();
    g.checkout_branch("same".into()).unwrap();
    assert!(g.finish_state_restore(outcome).is_err());
    g.checkout_branch("main".into()).unwrap();
    let outcome = g
        .prepare_state_restore(old, current, M::Tracked)
        .unwrap()
        .execute()
        .unwrap();
    g.add_node().unwrap();
    assert!(g.finish_state_restore(outcome).is_err());
    // Retained target on a sibling can be collected between capture and acceptance.
    let mut h = Graph::new();
    h.add_node().unwrap();
    let root = h.current_state_id();
    h.create_branch("discard".into()).unwrap();
    h.checkout_branch("discard".into()).unwrap();
    h.add_node().unwrap();
    let target = h.current_state_id();
    h.checkout_branch("main".into()).unwrap();
    let outcome = h
        .prepare_state_restore(target, root, M::Tracked)
        .unwrap()
        .execute()
        .unwrap();
    h.prune_branch("discard").unwrap();
    h.purge_unreferenced().unwrap();
    assert!(h.finish_state_restore(outcome).is_err());
    assert_eq!(h.current_state_id(), root);
}
#[test]
fn explicit_identical_tracked_restore_records_action_while_switch_is_reference_only() {
    let mut g = Graph::new();
    g.add_node().unwrap();
    let old = g.current_state_id();
    assert_eq!(g.restore_state(old, old, M::Switch).unwrap(), old);
    let id = g.restore_state(old, old, M::Tracked).unwrap();
    assert!(id > old);
    assert_eq!(g.history_records().last().unwrap().parents, vec![old]);
    assert!(g
        .publication_metadata(id)
        .unwrap()
        .unwrap()
        .change_times
        .is_empty());
    let outcome = g
        .prepare_state_restore(old, id, M::Switch)
        .unwrap()
        .execute()
        .unwrap();
    assert!(g
        .finish_state_restore_with_options(
            outcome,
            GraphMergeOptions {
                effective_at: Some(1),
                ..Default::default()
            },
            "switch".into(),
            "test".into(),
            None
        )
        .is_err());
    assert_eq!(g.current_state_id(), id);
    assert_eq!(g.restore_state(old, id, M::Switch).unwrap(), old);
    assert!(g.snapshot_at_commit(id).is_ok());
}
