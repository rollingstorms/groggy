use groggy::{
    state::{SnapshotMergeConflict as C, SnapshotMergeResolution as R},
    AttrValue as V, BatchNodeRef as N, ChangeKey as K, Graph, GraphBatch, GraphMergeOptions,
};
fn fixture() -> (Graph, u64, u64, u64) {
    let mut g = Graph::new();
    let root = g
        .apply_batch(
            GraphBatch {
                add_nodes: 2,
                add_edges: vec![(N::Added(0), N::Added(1))],
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "root".into(),
            "test".into(),
        )
        .unwrap()
        .state_id;
    g.create_branch("right".into()).unwrap();
    let left = g
        .writer("test")
        .with_effective_at(Some(20))
        .set_node_attr(0, "left".into(), V::Bool(true))
        .unwrap();
    g.checkout_branch("right".into()).unwrap();
    let right = g
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                add_edges: vec![(N::Existing(0), N::Added(0))],
                edge_endpoints: vec![(0, N::Added(0), N::Existing(1))],
                node_attrs: [("right".into(), vec![(N::Added(0), V::Null)])].into(),
                effective_at: Some(30),
                ..Default::default()
            },
            root,
            "right".into(),
            "test".into(),
        )
        .unwrap()
        .state_id;
    g.checkout_branch("main".into()).unwrap();
    (g, root, left, right)
}
#[test]
fn one_merge_publishes_exact_first_parent_delta_restored_ids_topology_and_clocks() {
    let (mut g, _, left, right) = fixture();
    let pin = g.read_view().unwrap();
    let versions = g.history_records().len();
    let outcome = g
        .prepare_merge_read(right, None, left)
        .unwrap()
        .execute()
        .unwrap()
        .into_outcome()
        .unwrap();
    assert_eq!(outcome.changes().nodes_added, vec![2]);
    assert_eq!(outcome.changes().edges_added, vec![(1, 0, 2)]);
    assert_eq!(outcome.changes().edge_endpoint_changes.len(), 1);
    let id = g
        .finish_merge(
            outcome,
            GraphMergeOptions {
                effective_at: Some(40),
                change_times: [(K::NodeAdded(2), 35)].into(),
            },
            "merge".into(),
            "test".into(),
        )
        .unwrap();
    assert!(id > right);
    assert_eq!(g.history_records().len(), versions + 1);
    let record = g
        .history_records()
        .into_iter()
        .find(|c| c.id == id)
        .unwrap();
    assert_eq!(record.parents, vec![left, right]);
    assert_eq!(g.current_state_id(), id);
    assert!(g.has_node(2));
    assert_eq!(g.edge_endpoints(0).unwrap(), (2, 1));
    assert_eq!(
        g.get_node_attr(0, &"left".into()).unwrap(),
        Some(V::Bool(true))
    );
    assert_eq!(g.get_node_attr(2, &"right".into()).unwrap(), Some(V::Null));
    assert!(!pin.contains_node(2));
    assert_eq!(pin.edge_endpoints(0).unwrap(), (0, 1));
    assert!(!g.read_view_at(34, None).unwrap().contains_node(2));
    assert!(g.read_view_at(35, None).unwrap().contains_node(2));
    assert_eq!(
        g.read_view_at(40, None).unwrap().edge_endpoints(0).unwrap(),
        (2, 1)
    );
    let cutoff = g.publication_metadata(left).unwrap().unwrap().recorded_at;
    assert!(!g.read_view_at(100, Some(cutoff)).unwrap().contains_node(2));
    let clock = g.publication_metadata(id).unwrap().unwrap();
    assert_eq!(clock.change_times[&K::NodeAttribute(2, "right".into())], 35);
    let restored = g.owned_snapshot_at_commit(id).unwrap().to_graph().unwrap();
    assert_eq!(restored.edge_endpoints(0).unwrap(), (2, 1));
    g.check_integrity().unwrap();
    let next = g.add_node().unwrap();
    assert_eq!(next, 3);
    g.checkout_branch("right".into()).unwrap();
    assert_eq!(g.current_state_id(), right);
    assert!(g.get_node_attr(0, &"left".into()).unwrap().is_none());
}
#[test]
fn invalid_timeline_and_unrelated_time_override_reject_without_mutation_or_id_consumption() {
    let (mut g, _, left, right) = fixture();
    let versions = g.history_records().len();
    for options in [
        GraphMergeOptions {
            effective_at: Some(5),
            ..Default::default()
        },
        GraphMergeOptions {
            effective_at: Some(40),
            change_times: [(K::NodeAttribute(999, "missing".into()), 40)].into(),
        },
    ] {
        let outcome = g
            .prepare_merge_read(right, None, left)
            .unwrap()
            .execute()
            .unwrap()
            .into_outcome()
            .unwrap();
        assert!(g
            .finish_merge(outcome, options, "bad".into(), "test".into())
            .is_err());
        assert_eq!(g.current_state_id(), left);
        assert!(!g.has_node(2));
        assert_eq!(g.history_records().len(), versions);
    }
    let outcome = g
        .prepare_merge_read(right, None, left)
        .unwrap()
        .execute()
        .unwrap()
        .into_outcome()
        .unwrap();
    let id = g
        .finish_merge(
            outcome,
            GraphMergeOptions {
                effective_at: Some(40),
                ..Default::default()
            },
            "good".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(id, right + 1);
}
#[test]
fn exact_float_resolution_and_data_identical_merges_record_without_checkpoint_duplication() {
    let mut g = Graph::new();
    g.add_node().unwrap();
    g.set_node_attr(0, "v".into(), V::Float64(1.0)).unwrap();
    g.create_branch("right".into()).unwrap();
    g.set_node_attr(0, "v".into(), V::Float64(0.0)).unwrap();
    let left = g.current_state_id();
    g.checkout_branch("right".into()).unwrap();
    g.set_node_attr(0, "v".into(), V::Float64(-0.0)).unwrap();
    let right = g.current_state_id();
    g.checkout_branch("main".into()).unwrap();
    let outcome = g
        .prepare_merge_read(right, None, left)
        .unwrap()
        .execute()
        .unwrap()
        .resolve_with(|c| match c {
            C::NodeProperty { right, .. } => Ok(R::Property(right.clone())),
            _ => panic!(),
        })
        .unwrap();
    assert_eq!(outcome.changes().node_attr_changes.len(), 1);
    let id = g
        .finish_merge(
            outcome,
            GraphMergeOptions::default(),
            "float".into(),
            "test".into(),
        )
        .unwrap();
    if let Some(V::Float64(v)) = g.get_node_attr(0, &"v".into()).unwrap() {
        assert_eq!(v.to_bits(), (-0.0_f64).to_bits());
    } else {
        panic!()
    }
    let n = g.history_records().len();
    assert_eq!(g.commit("label".into(), "test".into()).unwrap(), id);
    assert_eq!(g.history_records().len(), n);
    let outcome = g
        .prepare_merge_read(right, None, id)
        .unwrap()
        .execute()
        .unwrap()
        .into_outcome()
        .unwrap();
    assert!(outcome.changes().is_empty());
    let next = g
        .finish_merge(
            outcome,
            GraphMergeOptions::default(),
            "join".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(next, id + 1);
    g.check_integrity().unwrap();
}

#[test]
fn native_manual_writer_merge_uses_session_author_time_and_preserves_manual_policy() {
    let mut g = Graph::new_manual();
    let root = g
        .writer("test")
        .with_effective_at(Some(10))
        .add_node(Default::default())
        .unwrap();
    assert_eq!(root, 0);
    g.create_branch("right".into()).unwrap();
    g.writer("test")
        .with_effective_at(Some(20))
        .set_node_attr(0, "left".into(), V::Int(1))
        .unwrap();
    let left = g.current_state_id();
    g.checkout_branch("right".into()).unwrap();
    g.writer("test")
        .with_effective_at(Some(30))
        .set_node_attr(0, "right".into(), V::Int(2))
        .unwrap();
    let right = g.current_state_id();
    g.checkout_branch("main".into()).unwrap();
    let mut writer = g.writer("merge-session").with_effective_at(Some(40));
    let outcome = writer
        .prepare_merge(right, None)
        .unwrap()
        .execute()
        .unwrap()
        .into_outcome()
        .unwrap();
    let id = writer
        .finish_merge(outcome, GraphMergeOptions::default(), "session")
        .unwrap();
    drop(writer);
    assert!(!g.automatic_publication());
    assert!(!g.has_uncommitted_changes());
    let record = g
        .history_records()
        .into_iter()
        .find(|c| c.id == id)
        .unwrap();
    assert_eq!(record.author, "merge-session");
    assert_eq!(record.parents, vec![left, right]);
    assert_eq!(record.publication.as_ref().unwrap().effective_at, 40);
    g.set_node_attr(0, "pending".into(), V::Int(1)).unwrap();
    assert!(g.has_uncommitted_changes());
}

#[test]
fn published_merges_preserve_parallel_fact_ids_loops_and_external_edge_names() {
    let mut g = Graph::new();
    g.add_nodes(2).unwrap();
    g.add_edge(0, 1).unwrap();
    g.create_branch("right".into()).unwrap();
    let loop_id = g.add_edge(0, 0).unwrap();
    let left = g.current_state_id();
    g.checkout_branch("right".into()).unwrap();
    let parallel = g.add_edge(0, 1).unwrap();
    g.writer("test")
        .set_edge_external_id(parallel, "parallel-fact".into())
        .unwrap();
    let right = g.current_state_id();
    g.checkout_branch("main".into()).unwrap();
    let outcome = g
        .prepare_merge_read(right, None, left)
        .unwrap()
        .execute()
        .unwrap()
        .into_outcome()
        .unwrap();
    assert_eq!(outcome.changes().edges_added, vec![(parallel, 0, 1)]);
    let id = g
        .finish_merge(
            outcome,
            GraphMergeOptions::default(),
            "parallel".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(g.edge_count(), 3);
    assert_eq!(g.edge_endpoints(loop_id).unwrap(), (0, 0));
    assert_eq!(g.edge_endpoints(parallel).unwrap(), (0, 1));
    assert_eq!(g.edge_by_external_id("parallel-fact", None), Some(parallel));
    let restored = g.prepare_archive().unwrap().restore().unwrap();
    assert_eq!(restored.current_state_id(), id);
    assert_eq!(restored.edge_count(), 3);
    restored.check_integrity().unwrap();
}
