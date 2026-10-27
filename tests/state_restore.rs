use groggy::{AttrValue as V, Graph, GraphSchema, StateRestoreOutcome, StateRestorePlan};
fn assert_send_sync<T: Send + Sync>() {}
#[test]
fn restore_retained_state_recovers_topology_properties_names_groups_and_keeps_future_history() {
    assert_send_sync::<StateRestorePlan>();
    assert_send_sync::<StateRestoreOutcome>();
    let mut g = Graph::new();
    g.add_nodes(2).unwrap();
    g.add_edge(0, 1).unwrap();
    g.writer("test")
        .set_node_external_id(0, "old".into())
        .unwrap();
    g.set_node_attr(0, "value".into(), V::Float64Vec(vec![1.0, 2.0]))
        .unwrap();
    let source = g.read_view().unwrap();
    let group = g.retain_group(&source, "source".into()).unwrap();
    let meta = g.create_meta_node(group).unwrap();
    let target = g.current_state_id();
    let original_clock = g.publication_metadata(target).unwrap();
    g.writer("test")
        .set_node_external_id(0, "new".into())
        .unwrap();
    let extra = g.add_node().unwrap();
    g.add_edge(0, extra).unwrap();
    g.set_edge_endpoints(0, extra, 1).unwrap();
    g.set_node_attr(0, "value".into(), V::Text("future".into()))
        .unwrap();
    g.remove_node(1).unwrap();
    let future = g.current_state_id();
    g.create_branch("future".into()).unwrap();
    let future_pin = g.read_view().unwrap();
    let versions = g.history_records().len();
    let allocation = g.check_integrity().unwrap();
    let plan = g
        .prepare_state_restore(target, future, groggy::StateRestoreMode::Switch)
        .unwrap();
    let outcome = std::thread::spawn(move || plan.execute().unwrap())
        .join()
        .unwrap();
    // Maintenance does not invalidate a logical target or captured column values.
    g.compact_pool_with_reclamation(true).unwrap();
    assert_eq!(g.finish_state_restore(outcome).unwrap(), target);
    assert_eq!(g.history_records().len(), versions);
    assert_eq!(g.publication_metadata(target).unwrap(), original_clock);
    assert_eq!(g.edge_endpoints(0).unwrap(), (0, 1));
    assert_eq!(
        g.get_node_attr(0, &"value".into()).unwrap(),
        Some(V::Float64Vec(vec![1.0, 2.0]))
    );
    assert_eq!(g.node_by_external_id("old", None), Some(0));
    assert_eq!(g.node_by_external_id("new", None), None);
    assert_eq!(
        g.allocated_entity_for_external_id("new"),
        Some(groggy::ExternalEntityRef::Node(0))
    );
    assert!(!g.has_node(extra));
    assert!(g.is_node_allocated(extra));
    assert!(future_pin.nodes().contains(&extra));
    assert_eq!(
        g.read_view()
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .edge_endpoints(0)
            .unwrap(),
        (0, 1)
    );
    let report = g.check_integrity().unwrap();
    assert_eq!(report.allocated_nodes, allocation.allocated_nodes);
    assert_eq!(report.allocated_edges, allocation.allocated_edges);
    assert_eq!(
        g.snapshot_at_commit(future)
            .unwrap()
            .node_attr(0, &"value".into()),
        Some(V::Text("future".into()))
    );
    let copy = g.prepare_archive().unwrap().restore().unwrap();
    assert_eq!(copy.current_state_id(), target);
    assert!(copy.snapshot_at_commit(future).is_ok());
    assert_eq!(g.add_node().unwrap(), extra + 1);
    g.checkout_branch("future".into()).unwrap();
    assert_eq!(g.current_state_id(), future);
    assert_eq!(
        g.get_node_attr(0, &"value".into()).unwrap(),
        Some(V::Text("future".into()))
    );
}

#[test]
fn stale_foreign_switched_branch_and_collected_targets_reject_without_mutation() {
    let mut g = Graph::new();
    g.add_node().unwrap();
    let root = g.current_state_id();
    let outcome = g
        .prepare_state_restore(root, root, groggy::StateRestoreMode::Switch)
        .unwrap()
        .execute()
        .unwrap();
    let mut foreign = g.prepare_archive().unwrap().restore().unwrap();
    assert!(foreign.finish_state_restore(outcome).is_err());
    let outcome = g
        .prepare_state_restore(root, root, groggy::StateRestoreMode::Switch)
        .unwrap()
        .execute()
        .unwrap();
    g.create_branch("same".into()).unwrap();
    g.checkout_branch("same".into()).unwrap();
    assert!(g.finish_state_restore(outcome).is_err());
    g.checkout_branch("main".into()).unwrap();
    g.create_branch("discard".into()).unwrap();
    g.checkout_branch("discard".into()).unwrap();
    let node = g.add_node().unwrap();
    let target = g.current_state_id();
    g.checkout_branch("main".into()).unwrap();
    let outcome = g
        .prepare_state_restore(target, root, groggy::StateRestoreMode::Switch)
        .unwrap()
        .execute()
        .unwrap();
    g.prune_branch("discard").unwrap();
    g.purge_unreferenced().unwrap();
    assert!(g.finish_state_restore(outcome).is_err());
    assert_eq!(g.current_state_id(), root);
    assert!(!g.is_node_allocated(node));
    let outcome = g
        .prepare_state_restore(root, root, groggy::StateRestoreMode::Switch)
        .unwrap()
        .execute()
        .unwrap();
    g.add_node().unwrap();
    assert!(g.finish_state_restore(outcome).is_err());
    assert!(g
        .prepare_state_restore(0, g.current_state_id(), groggy::StateRestoreMode::Switch)
        .is_err());
    assert!(g
        .prepare_state_restore(999, g.current_state_id(), groggy::StateRestoreMode::Switch)
        .is_err());
    g.check_integrity().unwrap();
}

#[test]
fn dirty_manual_state_requires_explicit_pending_rollback_and_schema_is_checked_before_restore() {
    let mut manual = Graph::new_manual();
    manual.add_node().unwrap();
    let root = manual.commit("root".into(), "test".into()).unwrap();
    manual
        .set_node_attr(0, "pending".into(), V::Int(9))
        .unwrap();
    assert!(manual
        .restore_state(root, root, groggy::StateRestoreMode::Switch)
        .is_err());
    assert_eq!(
        manual.get_node_attr(0, &"pending".into()).unwrap(),
        Some(V::Int(9))
    );
    manual.reset_hard().unwrap();
    assert_eq!(
        manual
            .restore_state(root, root, groggy::StateRestoreMode::Switch)
            .unwrap(),
        root
    );
    assert!(!manual.automatic_publication());
    let mut g = Graph::new();
    g.add_node().unwrap();
    g.set_node_labels(0, vec!["Entity".into()]).unwrap();
    g.set_node_attr(0, "value".into(), V::Text("old".into()))
        .unwrap();
    let target = g.current_state_id();
    g.set_node_attr(0, "value".into(), V::Int(1)).unwrap();
    let current = g.current_state_id();
    let schema =
        GraphSchema::from_json(r#"{"node_labels":{"Entity":{"value":{"type":"int"}}}}"#).unwrap();
    assert!(g
        .writer("test")
        .with_schema(&schema)
        .restore_state(target, groggy::StateRestoreMode::Switch)
        .is_err());
    assert_eq!(g.current_state_id(), current);
    assert_eq!(
        g.get_node_attr(0, &"value".into()).unwrap(),
        Some(V::Int(1))
    );
}
