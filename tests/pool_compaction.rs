use groggy::{AttrValue as V, Graph};

#[test]
fn compaction_remaps_selected_columns_without_losing_history_groups_inactive_values_or_ids() {
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 1).unwrap();
    graph
        .writer("test")
        .set_node_external_id(0, "reserved".into())
        .unwrap();
    let mut states = Vec::new();
    for i in 0..12 {
        graph.set_node_attr(0, "x".into(), V::Int(i % 3)).unwrap();
        graph.set_node_attr(2, "x".into(), V::Int(99)).unwrap();
        graph.set_edge_attr(0, "x".into(), V::Int(i % 2)).unwrap();
        states.push((graph.current_state_id(), i % 3, i % 2));
    }
    graph.create_branch("keep".into()).unwrap();
    let group = graph
        .retain_group(&graph.read_view().unwrap(), "frozen".into())
        .unwrap();
    let meta = graph.create_meta_node(group).unwrap();
    graph.remove_node(2).unwrap();
    let pin = graph.read_view().unwrap();
    let state = graph.current_state_id();
    let versions = graph.history_records().len();
    let before = graph.check_integrity().unwrap();
    let outcome = std::thread::spawn({
        let plan = graph.prepare_pool_compaction().unwrap();
        move || plan.execute().unwrap()
    })
    .join()
    .unwrap();
    let report = graph.finish_pool_compaction(outcome).unwrap();
    assert_eq!(report.selected_state, state);
    assert!(report.current_node_values < report.previous_node_values);
    assert!(report.current_edge_values < report.previous_edge_values);
    assert_eq!(report.current_edge_values, 2);
    assert_eq!(
        graph.get_node_attr(0, &"x".into()).unwrap(),
        Some(V::Int(2))
    );
    assert_eq!(
        graph.get_edge_attr(0, &"x".into()).unwrap(),
        Some(V::Int(1))
    );
    assert_eq!(pin.node_attr(0, &"x".into()).unwrap(), Some(V::Int(2)));
    assert!(graph.has_node(meta));
    let after = graph.check_integrity().unwrap();
    assert_eq!(after.retained_versions, versions);
    assert_eq!(after.allocated_nodes, before.allocated_nodes);
    assert_eq!(
        after.external_id_reservations,
        before.external_id_reservations
    );
    assert_eq!(
        after.retained_group_references,
        before.retained_group_references
    );
    // The inactive node's distinct raw value is still present, even before replay.
    assert!(graph
        .pool()
        .get_node_attribute_column(&"x".into())
        .unwrap()
        .as_slice()
        .contains(&V::Int(99)));
    let second = graph.compact_pool().unwrap();
    assert_eq!(second.current_node_values, second.previous_node_values);
    assert_eq!(second.current_edge_values, second.previous_edge_values);
    let mut copy = graph.prepare_archive().unwrap().restore().unwrap();
    for (version, node, edge) in states {
        let snapshot = copy.snapshot_at_commit(version).unwrap();
        assert_eq!(snapshot.node_attr(0, &"x".into()), Some(V::Int(node)));
        assert_eq!(snapshot.node_attr(2, &"x".into()), Some(V::Int(99)));
        assert_eq!(snapshot.edge_attr(0, &"x".into()), Some(V::Int(edge)));
    }
    copy.checkout_branch("keep".into()).unwrap();
    assert!(copy.has_node(2));
    assert_eq!(
        copy.get_node_attr(2, &"x".into()).unwrap(),
        Some(V::Int(99))
    );
    copy.check_integrity().unwrap();
    assert_eq!(copy.add_node().unwrap(), meta + 1);
}

#[test]
fn deduplication_preserves_exact_float_bits_json_spelling_and_null_absence() {
    let mut graph = Graph::new();
    graph.add_nodes(9).unwrap();
    let values = [
        V::Float64(-0.0),
        V::Float64(0.0),
        V::Float64(f64::from_bits(0x7ff8000000000001)),
        V::Float64(f64::from_bits(0x7ff800000000007b)),
        V::Json("{\"x\":1}".into()),
        V::Json("{ \"x\": 1 }".into()),
        V::Null,
        V::Text("null".into()),
    ];
    for _ in 0..3 {
        for (id, value) in values.iter().enumerate() {
            graph.set_node_attr(id, "v".into(), value.clone()).unwrap();
        }
    }
    let report = graph.compact_pool().unwrap();
    assert_eq!(
        (report.previous_node_values, report.current_node_values),
        (33, 9)
    );
    for (id, value) in values.iter().enumerate() {
        let actual = graph.get_node_attr(id, &"v".into()).unwrap().unwrap();
        match (value, actual) {
            (V::Float64(a), V::Float64(b)) => assert_eq!(a.to_bits(), b.to_bits()),
            (a, b) => assert_eq!(a, &b),
        }
    }
    assert_eq!(graph.get_node_attr(8, &"v".into()).unwrap(), None);
    graph.optimize().unwrap();
    graph.check_integrity().unwrap();
}

#[test]
fn stale_foreign_and_dirty_compactions_reject_without_changing_storage() {
    let mut graph = Graph::new();
    graph.add_node().unwrap();
    graph.set_node_attr(0, "x".into(), V::Int(1)).unwrap();
    let plan = graph.prepare_pool_compaction().unwrap();
    graph.set_node_attr(0, "x".into(), V::Int(2)).unwrap();
    let before = graph.check_integrity().unwrap();
    assert!(graph
        .finish_pool_compaction(plan.execute().unwrap())
        .is_err());
    assert_eq!(graph.check_integrity().unwrap(), before);
    let foreign = graph.prepare_pool_compaction().unwrap().execute().unwrap();
    let mut other = graph.prepare_archive().unwrap().restore().unwrap();
    assert!(other.finish_pool_compaction(foreign).is_err());
    let mut manual = Graph::new_manual();
    manual.add_node().unwrap();
    assert!(manual.prepare_pool_compaction().is_err());
    assert!(manual.optimize().is_err());
}

#[test]
fn explicit_reclamation_discards_only_values_without_any_retained_history_or_working_root() {
    use groggy::ExternalEntityRef;
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 1).unwrap();
    graph
        .set_node_attr(2, "inactive".into(), V::Int(99))
        .unwrap();
    graph.remove_node(2).unwrap();
    graph.create_branch("discard".into()).unwrap();
    graph.checkout_branch("discard".into()).unwrap();
    graph
        .set_node_attr(0, "ghost".into(), V::Float64(-0.0))
        .unwrap();
    graph
        .set_edge_attr(0, "ghost".into(), V::Text("edge orphan".into()))
        .unwrap();
    graph
        .writer("test")
        .set_node_external_id(0, "reserved ghost".into())
        .unwrap();
    let gone = graph.current_state_id();
    let pin = graph.read_view().unwrap();
    let owned = graph.owned_snapshot_at_commit(gone).unwrap();
    let group = graph.retain_group(&pin, "frozen ghost".into()).unwrap();
    graph.checkout_branch("main".into()).unwrap();
    let meta = graph.create_meta_node(group).unwrap();
    // All versions, including non-selected and unreferenced ones, protect raw values.
    let retained = graph.compact_pool_with_reclamation(true).unwrap();
    assert_eq!(retained.reclaimed_unreferenced_node_values, 0);
    assert_eq!(retained.reclaimed_unreferenced_edge_values, 0);
    graph.prune_branch("discard").unwrap();
    assert!(graph.snapshot_at_commit(gone).is_err());
    graph.compact_pool().unwrap();
    assert_eq!(
        graph
            .pool()
            .get_node_attribute_column(&"ghost".into())
            .unwrap()
            .len(),
        1
    );
    let report = graph.compact_pool_with_reclamation(true).unwrap();
    assert!(report.reclaimed_unreferenced_node_values >= 2);
    assert_eq!(report.reclaimed_unreferenced_edge_values, 1);
    assert_eq!(
        graph
            .pool()
            .get_node_attribute_column(&"ghost".into())
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        graph
            .pool()
            .get_edge_attribute_column(&"ghost".into())
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        graph
            .pool()
            .get_node_attribute_column(&"inactive".into())
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        graph.allocated_entity_for_external_id("reserved ghost"),
        Some(ExternalEntityRef::Node(0))
    );
    let group_view = graph.read_view().unwrap().meta_subgraph(meta).unwrap();
    assert!(
        matches!(group_view.node_attr(0,&"ghost".into()).unwrap(),Some(V::Float64(x)) if x.to_bits()==(-0.0f64).to_bits())
    );
    assert_eq!(
        pin.edge_attr(0, &"ghost".into()).unwrap(),
        Some(V::Text("edge orphan".into()))
    );
    assert!(owned.to_graph().unwrap().snapshot_at_commit(gone).is_ok());
    let again = graph.compact_pool_with_reclamation(true).unwrap();
    assert_eq!(again.reclaimed_unreferenced_node_values, 0);
    assert_eq!(again.reclaimed_unreferenced_edge_values, 0);
    graph.check_integrity().unwrap();
    let mut copy = graph.prepare_archive().unwrap().restore().unwrap();
    assert_eq!(copy.add_node().unwrap(), meta + 1);
    copy.check_integrity().unwrap();
}

#[test]
fn explicit_reclamation_collects_discarded_manual_values_and_rejects_changed_history_roots() {
    let mut graph = Graph::new_manual();
    graph.add_node().unwrap();
    graph.set_node_attr(0, "x".into(), V::Int(1)).unwrap();
    graph.commit("root".into(), "test".into()).unwrap();
    graph.set_node_attr(0, "x".into(), V::Int(2)).unwrap();
    let pin = graph.read_view().unwrap();
    graph.reset_hard().unwrap();
    let report = graph.compact_pool_with_reclamation(true).unwrap();
    assert_eq!(report.reclaimed_unreferenced_node_values, 1);
    assert_eq!(pin.node_attr(0, &"x".into()).unwrap(), Some(V::Int(2)));
    assert_eq!(
        graph.get_node_attr(0, &"x".into()).unwrap(),
        Some(V::Int(1))
    );
    graph.create_branch("discard".into()).unwrap();
    graph.checkout_branch("discard".into()).unwrap();
    graph.add_node().unwrap();
    graph.commit("other".into(), "test".into()).unwrap();
    graph.checkout_branch("main".into()).unwrap();
    let outcome = graph
        .prepare_pool_compaction_with_reclamation(true)
        .unwrap()
        .execute()
        .unwrap();
    graph.prune_branch("discard").unwrap();
    // Root-set changes invalidate a detached reclamation even with unchanged columns.
    assert!(graph.finish_pool_compaction(outcome).is_err());
    graph.compact_pool_with_reclamation(true).unwrap();
    graph.check_integrity().unwrap();
}

#[test]
fn unreferenced_retained_versions_protect_values_until_explicit_history_collection() {
    let mut graph = Graph::new();
    graph.add_node().unwrap();
    let root = graph.current_state_id();
    graph
        .set_node_attr(0, "future".into(), V::Float64(-0.0))
        .unwrap();
    let future = graph.current_state_id();
    let mut past = graph
        .owned_snapshot_at_commit(root)
        .unwrap()
        .to_graph()
        .unwrap();
    assert_eq!(past.current_state_id(), root);
    let report = past.compact_pool_with_reclamation(true).unwrap();
    assert_eq!(report.reclaimed_unreferenced_node_values, 0);
    assert!(past.snapshot_at_commit(future).is_ok());
    let raw = past.pool();
    assert!(
        matches!(raw.get_node_attribute_column(&"future".into()).unwrap().get(0),Some(V::Float64(v)) if v.to_bits()==(-0.0f64).to_bits())
    );
    drop(raw);
    assert_eq!(past.gc_history().unwrap(), 1);
    assert_eq!(
        past.compact_pool_with_reclamation(true)
            .unwrap()
            .reclaimed_unreferenced_node_values,
        1
    );
    past.check_integrity().unwrap();
}
