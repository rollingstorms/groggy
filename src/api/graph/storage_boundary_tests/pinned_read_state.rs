use crate::{state::PinnedReadState, types::GraphType, AttrValue, Graph};

fn shares_payload(a: &PinnedReadState, b: &PinnedReadState) -> bool {
    std::ptr::eq(a.snapshot(), b.snapshot())
}

#[test]
fn working_capture_includes_uncheckpointed_changes_without_recording_history() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    graph
        .set_node_attr(node, "value".into(), AttrValue::Int(i64::MAX))
        .unwrap();
    let checkpoint = graph.commit("base".into(), "test".into()).unwrap();
    let first = graph.pin_current_state().unwrap();
    let topology_version = graph.space().get_version();
    graph
        .set_node_attr(node, "value".into(), AttrValue::Null)
        .unwrap();
    assert_eq!(topology_version, graph.space().get_version());
    let history_count = graph.history_records().len();
    let second = graph.pin_current_state().unwrap();
    assert_eq!(graph.current_state_id(), checkpoint);
    assert_eq!(graph.history_records().len(), history_count);
    assert_eq!(
        first.node_attr(node, &"value".into()),
        Some(AttrValue::Int(i64::MAX))
    );
    assert_eq!(
        second.node_attr(node, &"value".into()),
        Some(AttrValue::Null)
    );
    assert_eq!(second.node_attr(node, &"absent".into()), None);
    let cloned = second.clone();
    assert!(std::ptr::eq(second.snapshot(), cloned.snapshot()));
    graph.remove_node(node).unwrap();
    assert!(second.existence().contains_node(node));
    assert!(!graph
        .pin_current_state()
        .unwrap()
        .existence()
        .contains_node(node));
    drop(graph);
    assert_eq!(
        second.node_attr(node, &"value".into()),
        Some(AttrValue::Null)
    );
}
#[test]
fn read_topology_preserves_direction_parallel_occurrences_isolates_and_self_loops() {
    for directed in [false, true] {
        let mut graph = Graph::new_with_type_manual(if directed {
            GraphType::Directed
        } else {
            GraphType::Undirected
        });
        let nodes = graph.add_nodes(3).unwrap();
        let a = nodes[0];
        let b = nodes[1];
        let edge = graph.add_edge(a, b).unwrap();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(a, a).unwrap();
        graph
            .set_edge_attr(edge, "weight".into(), AttrValue::Int(3))
            .unwrap();
        let pin = graph.pin_current_state().unwrap();
        assert_eq!(pin.neighbors(a).unwrap(), &vec![a, b, b]);
        assert_eq!(
            pin.neighbors(b).unwrap(),
            &if directed { vec![] } else { vec![a, a] }
        );
        assert!(pin.neighbors(nodes[2]).unwrap().is_empty());
        assert!(pin.neighbors(usize::MAX).is_none());
        graph.remove_edge(edge).unwrap();
        assert_eq!(
            pin.edge_attr(edge, &"weight".into()),
            Some(AttrValue::Int(3))
        );
        assert_eq!(pin.snapshot().edges.get(&edge), Some(&(a, b)));
    }
}
#[test]
fn temporal_facade_reuses_shared_state_without_duplicate_property_maps() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    graph
        .set_node_attr(node, "value".into(), AttrValue::Int(7))
        .unwrap();
    let id = graph.commit("first".into(), "test".into()).unwrap();
    let temporal = graph.snapshot_at_commit(id).unwrap();
    let clone = temporal.clone();
    assert!(std::ptr::eq(
        temporal.read_state().snapshot(),
        clone.read_state().snapshot()
    ));
    graph
        .set_node_attr(node, "value".into(), AttrValue::Int(8))
        .unwrap();
    assert_eq!(
        temporal.node_attr(node, &"value".into()),
        Some(AttrValue::Int(7))
    );
    assert_eq!(temporal.as_subgraph().unwrap().node_count(), 1);
}
#[test]
fn dangling_attribute_referents_fail_capture_without_silent_missing_values() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    graph
        .space_mut()
        .set_node_attr_index(node, "dangling".into(), usize::MAX);
    let count = graph.history_records().len();
    assert!(graph.pin_current_state().is_err());
    assert_eq!(graph.history_records().len(), count);
}
#[test]
fn frozen_read_payloads_can_cross_read_only_worker_boundaries() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<PinnedReadState>();
    let mut graph = Graph::new_manual();
    let id = graph.add_node().unwrap();
    let state = graph.pin_current_state().unwrap();
    assert!(
        std::thread::spawn(move || state.existence().contains_node(id))
            .join()
            .unwrap()
    );
}

#[test]
fn repeated_current_captures_share_payload_only_while_readers_hold_it() {
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    let first = graph.pin_current_state().unwrap();
    let second = graph.pin_current_state().unwrap();
    assert!(shares_payload(&first, &second));
    assert!(!first.shares_capture(&second));
    assert!(first.shares_capture(&first.clone()));
    let weak = first.downgrade_data();
    drop(first);
    assert!(weak.upgrade().is_some());
    drop(second);
    assert!(
        weak.upgrade().is_none(),
        "graph cache must not pin reader data"
    );
    let fresh = graph.pin_current_state().unwrap();
    assert_eq!(fresh.existence().nodes().len(), 3);
    assert!(weak.upgrade().is_none());
}

#[test]
fn warm_capture_rebinds_branch_and_does_not_reuse_old_write_provenance() {
    let mut graph = Graph::new();
    graph.add_node().unwrap();
    let first = graph.pin_current_state().unwrap();
    let first_branch = first.origin().unwrap().branch.clone();
    graph.create_branch("other".into()).unwrap();
    graph.checkout_branch("other".into()).unwrap();
    let other = graph.pin_current_state().unwrap();
    assert!(shares_payload(&first, &other));
    assert_eq!(first.origin().unwrap().branch, first_branch);
    assert_eq!(other.origin().unwrap().branch, "other");
    assert_eq!(other.write_state_id(), Some(graph.current_state_id()));
    assert!(other.read_time().is_none());
}

#[test]
fn warm_manual_capture_invalidates_for_property_unset_rewire_and_node_removal() {
    let mut graph = Graph::new_with_type_manual(GraphType::Directed);
    let nodes = graph.add_nodes(3).unwrap();
    let edge = graph.add_edge(nodes[0], nodes[1]).unwrap();
    graph
        .set_node_attr(nodes[0], "value".into(), AttrValue::Int(1))
        .unwrap();
    graph.commit("base".into(), "test".into()).unwrap();
    let initial = graph.pin_current_state().unwrap();
    let version = graph.space().get_version();
    graph
        .set_node_attr(nodes[0], "value".into(), AttrValue::Null)
        .unwrap();
    let changed = graph.pin_current_state().unwrap();
    assert_eq!(graph.space().get_version(), version);
    assert!(!shares_payload(&initial, &changed));
    assert!(changed.write_state_id().is_none());
    assert_eq!(
        initial.node_attr(nodes[0], &"value".into()),
        Some(AttrValue::Int(1))
    );
    assert_eq!(
        changed.node_attr(nodes[0], &"value".into()),
        Some(AttrValue::Null)
    );
    graph
        .space_mut()
        .unset_attr_index(nodes[0], &"value".into(), true);
    let unset = graph.pin_current_state().unwrap();
    assert!(!shares_payload(&unset, &changed));
    assert_eq!(unset.node_attr(nodes[0], &"value".into()), None);
    graph.set_edge_endpoints(edge, nodes[2], nodes[1]).unwrap();
    let rewired = graph.pin_current_state().unwrap();
    assert!(!shares_payload(&rewired, &unset));
    assert_eq!(rewired.snapshot().edges[&edge], (nodes[2], nodes[1]));
    assert_eq!(initial.snapshot().edges[&edge], (nodes[0], nodes[1]));
    assert!(rewired.neighbors(nodes[0]).unwrap().is_empty());
    graph.remove_node(nodes[2]).unwrap();
    let removed = graph.pin_current_state().unwrap();
    assert!(!removed.existence().contains_edge(edge));
    assert!(!removed.existence().contains_node(nodes[2]));
    assert!(rewired.existence().contains_edge(edge));
}

#[test]
fn warm_capture_does_not_hide_mapping_corruption_or_failed_batches() {
    let mut graph = Graph::new();
    let node = graph.add_node().unwrap();
    let first = graph.pin_current_state().unwrap();
    assert!(graph.set_edge_endpoints(999, node, node).is_err());
    assert!(shares_payload(&first, &graph.pin_current_state().unwrap()));
    graph
        .space_mut()
        .set_attr_index(node, "bad".into(), usize::MAX, true);
    assert!(graph.pin_current_state().is_err());
    graph
        .space_mut()
        .unset_attr_index(node, &"bad".into(), true);
    let recovered = graph.pin_current_state().unwrap();
    assert!(!shares_payload(&recovered, &first));
}

#[test]
fn pool_compaction_invalidates_reuse_without_changing_frozen_values() {
    let mut graph = Graph::new();
    let node = graph.add_node().unwrap();
    graph
        .set_node_attr(node, "value".into(), AttrValue::Int(1))
        .unwrap();
    graph
        .set_node_attr(node, "value".into(), AttrValue::Int(2))
        .unwrap();
    let before = graph.pin_current_state().unwrap();
    let head = graph.current_state_id();
    graph.compact_pool().unwrap();
    let after = graph.pin_current_state().unwrap();
    assert!(!shares_payload(&before, &after));
    assert_eq!(
        before.node_attr(node, &"value".into()),
        Some(AttrValue::Int(2))
    );
    assert_eq!(
        after.node_attr(node, &"value".into()),
        Some(AttrValue::Int(2))
    );
    assert_eq!(graph.current_state_id(), head);
}

#[test]
fn warm_capture_refreshes_group_catalog_without_requiring_a_graph_delta() {
    let mut graph = Graph::new();
    graph.add_node().unwrap();
    let before = graph.pin_current_state().unwrap();
    let prepared = graph
        .pool
        .borrow_mut()
        .prepare_frozen_group(crate::GraphView::all(before.clone()), "test".into())
        .unwrap();
    let id = prepared.id;
    graph.pool.borrow_mut().install_stored_subgraph(prepared);
    let after = graph.pin_current_state().unwrap();
    assert!(shares_payload(&before, &after));
    assert!(before.retained_group(id).is_err());
    assert!(after.retained_group(id).unwrap().contains_node(0));
}

#[test]
fn trusted_mutable_pool_access_invalidates_even_without_a_space_mapping_change() {
    let mut graph = Graph::new_with_type_manual(GraphType::Directed);
    graph.add_nodes(3).unwrap();
    let edge = graph.add_edge(0, 1).unwrap();
    let first = graph.pin_current_state().unwrap();
    graph.pool_mut().restore_edge_revision(edge, 2, 1).unwrap();
    let second = graph.pin_current_state().unwrap();
    assert!(!shares_payload(&first, &second));
    assert_eq!(second.snapshot().edges[&edge], (2, 1));
    assert_eq!(first.snapshot().edges[&edge], (0, 1));
}

#[test]
fn accepting_manual_changes_refreshes_state_identity_and_write_coordinates() {
    let mut graph = Graph::new_manual();
    graph.add_node().unwrap();
    let dirty = graph.pin_current_state().unwrap();
    assert!(dirty.write_state_id().is_none());
    let accepted = graph.commit("accept".into(), "test".into()).unwrap();
    let clean = graph.pin_current_state().unwrap();
    assert!(!shares_payload(&dirty, &clean));
    assert_eq!(clean.snapshot().state_id, accepted);
    assert_eq!(clean.write_state_id(), Some(accepted));
    assert!(dirty.write_state_id().is_none());
}
