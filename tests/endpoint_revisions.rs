use groggy::{AttrValue as V, BatchNodeRef as N, ChangeKey, Graph, GraphBatch};
fn fixture() -> (Graph, u64, usize) {
    let mut g = Graph::new();
    let b = GraphBatch {
        add_nodes: 3,
        add_edges: vec![(N::Added(0), N::Added(1))],
        effective_at: Some(10),
        ..Default::default()
    };
    let state = g
        .apply_batch(b, 0, "initial".into(), "test".into())
        .unwrap()
        .state_id;
    g.writer("test")
        .with_effective_at(Some(10))
        .set_edge_attr(0, "weight".into(), V::Int(7))
        .unwrap();
    (g, state, 0)
}
#[test]
fn endpoint_revisions_preserve_properties_identity_recorded_and_effective_views_and_restore() {
    let (mut g, first, id) = fixture();
    g.create_branch("before".into()).unwrap();
    assert_eq!(g.neighbors(0).unwrap(), vec![1]);
    let pin = g.read_view().unwrap();
    let state = g
        .writer("test")
        .with_effective_at(Some(20))
        .set_edge_endpoints(id, 2, 1)
        .unwrap();
    assert_eq!(g.edge_endpoints(id).unwrap(), (2, 1));
    assert!(g.neighbors(0).unwrap().is_empty());
    assert_eq!(g.neighbors(2).unwrap(), vec![1]);
    assert_eq!(
        g.read_view_at(19, None)
            .unwrap()
            .edge_endpoints(id)
            .unwrap(),
        (0, 1)
    );
    assert_eq!(
        g.read_view_at(20, None)
            .unwrap()
            .edge_endpoints(id)
            .unwrap(),
        (2, 1)
    );
    assert_eq!(pin.edge_endpoints(id).unwrap(), (0, 1));
    assert_eq!(
        g.get_edge_attr(id, &"weight".into()).unwrap(),
        Some(V::Int(7))
    );
    assert_eq!(
        g.diff_commits(first, state)
            .unwrap()
            .edge_endpoint_changes
            .len(),
        1
    );
    assert_eq!(
        g.publication_metadata(state).unwrap().unwrap().change_times[&ChangeKey::EdgeEndpoints(id)],
        20
    );
    g.check_integrity().unwrap();
    g.create_branch("after".into()).unwrap();
    let mut copy = g.prepare_archive().unwrap().restore().unwrap();
    copy.checkout_branch("before".into()).unwrap();
    assert_eq!(copy.edge_endpoints(id).unwrap(), (0, 1));
    copy.checkout_branch("after".into()).unwrap();
    assert_eq!(copy.edge_endpoints(id).unwrap(), (2, 1));
    copy.check_integrity().unwrap();
}
#[test]
fn malformed_duplicate_and_early_revisions_reject_without_any_allocation() {
    let (mut g, _, id) = fixture();
    for batch in [
        GraphBatch {
            add_nodes: 1,
            edge_endpoints: vec![(id, N::Existing(0), N::Existing(99))],
            ..Default::default()
        },
        GraphBatch {
            edge_endpoints: vec![(id, N::Existing(0), N::Existing(2)); 2],
            ..Default::default()
        },
        GraphBatch {
            edge_endpoints: vec![(id, N::Existing(0), N::Existing(2))],
            effective_at: Some(9),
            ..Default::default()
        },
    ] {
        let state = g.current_state_id();
        let before = g.check_integrity().unwrap();
        assert!(g
            .apply_batch(batch, state, "bad".into(), "test".into())
            .is_err());
        assert_eq!(g.current_state_id(), state);
        assert_eq!(g.check_integrity().unwrap(), before);
    }
}

#[test]
fn mixed_rewire_can_detach_a_removed_node_and_attach_a_new_node_in_one_version() {
    let (mut g, _, id) = fixture();
    let prior = g.current_state_id();
    let batch = GraphBatch {
        add_nodes: 1,
        edge_endpoints: vec![(id, N::Added(0), N::Existing(1))],
        remove_nodes: vec![N::Existing(0)],
        edge_attrs: std::collections::HashMap::from([(
            "weight".into(),
            vec![(groggy::BatchEdgeRef::Existing(id), V::Int(8))],
        )]),
        effective_at: Some(20),
        ..Default::default()
    };
    let state = g
        .apply_batch(batch, prior, "mixed".into(), "test".into())
        .unwrap()
        .state_id;
    assert!(!g.has_node(0));
    assert!(g.has_edge(id));
    assert_eq!(g.edge_endpoints(id).unwrap(), (3, 1));
    assert_eq!(
        g.get_edge_attr(id, &"weight".into()).unwrap(),
        Some(V::Int(8))
    );
    assert!(!g
        .publication_metadata(state)
        .unwrap()
        .unwrap()
        .change_times
        .contains_key(&ChangeKey::EdgeRemoved(id)));
    assert_eq!(
        g.read_view_at(20, None)
            .unwrap()
            .edge_endpoints(id)
            .unwrap(),
        (3, 1)
    );
    g.check_integrity().unwrap();
}

#[test]
fn recorded_index_and_temporal_diffs_follow_endpoint_revisions_without_resetting_properties() {
    let (mut g, _, id) = fixture();
    let before = g.current_state_id();
    let pin = g.snapshot_at_commit(before).unwrap();
    let after = g
        .writer("test")
        .with_effective_at(Some(20))
        .set_edge_endpoints(id, 2, 1)
        .unwrap();
    let index = g.build_temporal_index().unwrap();
    assert_eq!(index.neighbors_at_commit(0, before), vec![1]);
    assert!(index.neighbors_at_commit(0, after).is_empty());
    assert_eq!(index.neighbors_at_commit(2, after), vec![1]);
    assert_eq!(index.neighbors_in_window(1, before, after), vec![0, 2]);
    assert_eq!(index.edges_changed_in_commit(after), vec![id]);
    let delta =
        groggy::algorithms::TemporalDelta::compute(&pin, &g.snapshot_at_commit(after).unwrap())
            .unwrap();
    assert!(delta.edges_added.is_empty() && delta.edges_removed.is_empty());
    assert_eq!(delta.edge_endpoint_changes[0].old_endpoints, (0, 1));
    assert!(delta.affected_edges().contains(&id));
    assert_eq!(
        pin.read_state().snapshot().edges.get(&id).copied(),
        Some((0, 1))
    );
}

#[test]
fn endpoint_time_ties_and_backdated_removal_validation_share_the_topology_contract() {
    let (mut g, _, id) = fixture();
    g.writer("test")
        .with_effective_at(Some(20))
        .set_edge_endpoints(id, 2, 1)
        .unwrap();
    g.writer("test")
        .with_effective_at(Some(20))
        .set_edge_endpoints(id, 1, 2)
        .unwrap();
    assert_eq!(
        g.read_view_at(20, None)
            .unwrap()
            .edge_endpoints(id)
            .unwrap(),
        (1, 2)
    );
    let state = g.current_state_id();
    assert!(g
        .writer("test")
        .with_effective_at(Some(15))
        .remove_edges(vec![id])
        .is_err());
    assert_eq!(g.current_state_id(), state);
    assert!(g.has_edge(id));
}

#[test]
fn reversal_preserves_parallel_instances_loops_external_names_and_owned_group_sources() {
    let (mut g, _, id) = fixture();
    g.writer("test")
        .with_effective_at(Some(10))
        .set_edge_external_id(id, "fact".into())
        .unwrap();
    let source = g.read_view().unwrap();
    let group = g.retain_group(&source, "frozen".into()).unwrap();
    let meta = g.create_meta_node(group).unwrap();
    let parallel = g.add_edge(0, 1).unwrap();
    let loop_id = g.add_edge(2, 2).unwrap();
    g.reverse_edges(&[id, parallel, loop_id]).unwrap();
    assert_eq!(g.edge_endpoints(id).unwrap(), (1, 0));
    assert_eq!(g.edge_endpoints(parallel).unwrap(), (1, 0));
    assert_eq!(g.edge_endpoints(loop_id).unwrap(), (2, 2));
    assert_eq!(g.edge_by_external_id("fact", None), Some(id));
    assert_eq!(
        g.read_view()
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .edge_endpoints(id)
            .unwrap(),
        (0, 1)
    );
    let state = g.current_state_id();
    assert!(g.reverse_edges(&[id, id]).is_err());
    assert_eq!(g.current_state_id(), state);
    g.purge_unreferenced().unwrap();
    g.check_integrity().unwrap();
    let copy = g
        .owned_snapshot_at_commit(state)
        .unwrap()
        .to_graph()
        .unwrap();
    assert_eq!(copy.edge_endpoints(id).unwrap(), (1, 0));
}

#[test]
fn historical_revisions_protect_old_endpoint_nodes_until_the_branch_is_explicitly_pruned() {
    let (mut g, _, id) = fixture();
    g.create_branch("side".into()).unwrap();
    g.checkout_branch("side".into()).unwrap();
    let new = g.add_node().unwrap();
    g.set_edge_endpoints(id, new, 1).unwrap();
    let pin = g.read_view().unwrap();
    g.checkout_branch("main".into()).unwrap();
    assert_eq!(g.purge_unreferenced().unwrap().purged_nodes, 0);
    g.prune_branch("side").unwrap();
    assert_eq!(g.purge_unreferenced().unwrap().purged_nodes, 1);
    assert!(!g.is_node_allocated(new));
    assert_eq!(pin.edge_endpoints(id).unwrap(), (new, 1));
    g.check_integrity().unwrap();
}

#[test]
fn endpoint_tracker_coalesces_before_after_and_all_delta_forms_expose_changes() {
    let mut tracker = groggy::ChangeTracker::new();
    tracker
        .record_edge_endpoint_change(groggy::EdgeEndpointChange {
            edge_id: 7,
            old_endpoints: (0, 1),
            new_endpoints: (1, 2),
        })
        .unwrap();
    tracker
        .record_edge_endpoint_change(groggy::EdgeEndpointChange {
            edge_id: 7,
            old_endpoints: (1, 2),
            new_endpoints: (2, 0),
        })
        .unwrap();
    let changes = tracker.create_changeset();
    assert_eq!(changes.edge_endpoint_changes.len(), 1);
    assert_eq!(changes.edge_endpoint_changes[0].old_endpoints, (0, 1));
    assert_eq!(changes.edge_endpoint_changes[0].new_endpoints, (2, 0));
    assert!(!tracker.create_delta().is_empty());
    assert_eq!(tracker.create_delta().change_count(), 1);
    assert_eq!(tracker.change_summary().edge_endpoint_changes, 1);
    let empty = groggy::state::Delta::from_changes(groggy::state::change_tracker::ChangeSet::new());
    assert_ne!(
        groggy::state::Delta::from_changes(changes).content_hash,
        empty.content_hash
    );
    tracker.clear();
    assert!(!tracker.has_changes());
}

#[test]
fn ordinary_manual_endpoint_changes_coalesce_reset_and_new_edge_creation_consistently() {
    let mut g = Graph::new_manual();
    g.add_nodes(3).unwrap();
    let id = g.add_edge(0, 1).unwrap();
    g.set_edge_endpoints(id, 2, 1).unwrap();
    let initial = g.commit("initial".into(), "test".into()).unwrap();
    g.check_integrity().unwrap();
    assert_eq!(
        g.snapshot_at_commit(initial)
            .unwrap()
            .read_state()
            .snapshot()
            .edges[&id],
        (2, 1)
    );
    g.set_edge_endpoints(id, 1, 2).unwrap();
    g.set_edge_endpoints(id, 0, 2).unwrap();
    assert!(g.has_uncommitted_changes());
    g.reset_hard().unwrap();
    assert_eq!(g.edge_endpoints(id).unwrap(), (2, 1));
    assert!(!g.has_uncommitted_changes());
    g.set_edge_endpoints(id, 0, 1).unwrap();
    let changed = g.commit("changed".into(), "test".into()).unwrap();
    assert_eq!(
        g.diff_commits(initial, changed)
            .unwrap()
            .edge_endpoint_changes[0]
            .old_endpoints,
        (2, 1)
    );
    g.check_integrity().unwrap();
}

#[test]
fn schema_failure_in_a_mixed_rewire_batch_preserves_all_graph_state() {
    let (mut g, _, id) = fixture();
    g.set_edge_relationship_type(id, Some("LINK".into()))
        .unwrap();
    let schema = groggy::GraphSchema::from_json(
        r#"{"relationship_types":{"LINK":{"weight":{"type":"int"}}}}"#,
    )
    .unwrap();
    let before = g.check_integrity().unwrap();
    let batch = GraphBatch {
        edge_endpoints: vec![(id, N::Existing(2), N::Existing(1))],
        edge_attrs: std::collections::HashMap::from([(
            "weight".into(),
            vec![(
                groggy::BatchEdgeRef::Existing(id),
                V::Text("invalid".into()),
            )],
        )]),
        ..Default::default()
    };
    assert!(g
        .writer("test")
        .with_schema(&schema)
        .apply_batch(batch, "invalid")
        .is_err());
    assert_eq!(g.edge_endpoints(id).unwrap(), (0, 1));
    assert_eq!(g.check_integrity().unwrap(), before);
}

#[test]
fn directed_reversal_rebuilds_incoming_and_outgoing_cached_topology() {
    let mut g = Graph::with_config(groggy::GraphConfig {
        graph_type: groggy::types::GraphType::Directed,
        ..Default::default()
    })
    .unwrap();
    g.add_nodes(2).unwrap();
    g.add_edge(0, 1).unwrap();
    assert_eq!(g.out_degree(0).unwrap(), 1);
    assert_eq!(g.in_degree(0).unwrap(), 0);
    g.reverse_edges(&[0]).unwrap();
    assert_eq!(g.out_degree(0).unwrap(), 0);
    assert_eq!(g.in_degree(0).unwrap(), 1);
    assert_eq!(g.neighbors(1).unwrap(), vec![0]);
    g.set_edge_endpoints(0, 0, 0).unwrap();
    assert_eq!(g.out_degree(0).unwrap(), 1);
    assert_eq!(g.in_degree(0).unwrap(), 1);
    assert_eq!(g.neighbors(0).unwrap(), vec![0]);
    g.check_integrity().unwrap();
}

#[test]
fn recorded_index_rejects_invalid_endpoint_revision_targets_from_raw_history() {
    use groggy::state::change_tracker::ChangeSet;
    let mut history = groggy::HistoryForest::new();
    let root = history
        .create_commit(
            ChangeSet {
                nodes_added: vec![0, 1],
                edges_added: vec![(0, 0, 1)],
                ..ChangeSet::new()
            },
            "root".into(),
            "test".into(),
            None,
        )
        .unwrap();
    history
        .create_commit(
            ChangeSet {
                edge_endpoint_changes: vec![groggy::EdgeEndpointChange {
                    edge_id: 0,
                    old_endpoints: (0, 1),
                    new_endpoints: (99, 1),
                }],
                ..ChangeSet::new()
            },
            "invalid".into(),
            "test".into(),
            Some(root),
        )
        .unwrap();
    assert!(groggy::temporal::TemporalIndex::from_history(&history).is_err());
}

#[test]
fn older_owned_selection_restores_its_endpoints_and_retains_later_revisions_independently() {
    let (mut g, _, id) = fixture();
    let before = g.current_state_id();
    let after = g.set_edge_endpoints(id, 2, 1).unwrap();
    let owned = g.owned_snapshot_at_commit(before).unwrap();
    let mut copy = owned.to_graph().unwrap();
    assert_eq!(copy.edge_endpoints(id).unwrap(), (0, 1));
    assert_eq!(
        copy.snapshot_at_commit(after)
            .unwrap()
            .read_state()
            .snapshot()
            .edges[&id],
        (2, 1)
    );
    copy.set_edge_endpoints(id, 1, 0).unwrap();
    assert_eq!(g.edge_endpoints(id).unwrap(), (2, 1));
    assert_eq!(
        owned.to_graph().unwrap().edge_endpoints(id).unwrap(),
        (0, 1)
    );
    copy.check_integrity().unwrap();
}

#[test]
fn backdated_revisions_follow_effective_order_without_changing_recorded_before_values() {
    let (mut g, _, id) = fixture();
    let late = g
        .writer("test")
        .with_effective_at(Some(30))
        .set_edge_endpoints(id, 2, 1)
        .unwrap();
    let backdated = g
        .writer("test")
        .with_effective_at(Some(20))
        .set_edge_endpoints(id, 1, 0)
        .unwrap();
    assert_eq!(g.edge_endpoints(id).unwrap(), (1, 0));
    assert_eq!(
        g.read_view_at(25, None)
            .unwrap()
            .edge_endpoints(id)
            .unwrap(),
        (1, 0)
    );
    assert_eq!(
        g.read_view_at(35, None)
            .unwrap()
            .edge_endpoints(id)
            .unwrap(),
        (2, 1)
    );
    assert_eq!(
        g.diff_commits(late, backdated)
            .unwrap()
            .edge_endpoint_changes[0]
            .old_endpoints,
        (2, 1)
    );
    g.check_integrity().unwrap();
}
