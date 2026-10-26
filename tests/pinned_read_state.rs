use groggy::{state::PinnedReadState, types::GraphType, AttrValue, Graph};
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
