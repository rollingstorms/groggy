use groggy::{AttrValue as V, BatchChangeRef as T, BatchNodeRef as N, Graph, GraphBatch};
use std::collections::HashMap;
fn publish(graph: &mut Graph, batch: GraphBatch) -> u64 {
    graph
        .apply_batch(
            batch,
            graph.current_state_id(),
            "test".into(),
            "test".into(),
        )
        .unwrap()
        .state_id
}
fn create(graph: &mut Graph) -> u64 {
    publish(
        graph,
        GraphBatch {
            add_nodes: 2,
            add_edges: vec![(N::Added(0), N::Added(1))],
            effective_at: Some(10),
            node_attrs: HashMap::from([("value".into(), vec![(N::Added(0), V::Int(1))])]),
            ..Default::default()
        },
    )
}
fn update(graph: &mut Graph, at: i64, value: V) -> u64 {
    publish(
        graph,
        GraphBatch {
            effective_at: Some(at),
            node_attrs: HashMap::from([("value".into(), vec![(N::Existing(0), value)])]),
            ..Default::default()
        },
    )
}
#[test]
fn reconstructs_between_checkpoints_and_later_accepted_equal_time_correction_wins() {
    let mut graph = Graph::new_publishing();
    create(&mut graph);
    update(&mut graph, 30, V::Int(3));
    update(&mut graph, 20, V::Int(2));
    let captured = graph.read_view_at(25, None).unwrap();
    let head = graph.current_state_id();
    assert_eq!(
        captured.node_attr(0, &"value".into()).unwrap(),
        Some(V::Int(2))
    );
    assert_eq!(
        graph
            .read_view_at(35, None)
            .unwrap()
            .node_attr(0, &"value".into())
            .unwrap(),
        Some(V::Int(3))
    );
    update(&mut graph, 20, V::Null);
    let corrected = graph.read_view_at(25, None).unwrap();
    assert_eq!(
        corrected.node_attr(0, &"value".into()).unwrap(),
        Some(V::Null)
    );
    assert_eq!(corrected.node_attr(1, &"value".into()).unwrap(), None);
    assert_eq!(
        captured.read_state().read_time().unwrap().published_head,
        head
    );
    assert_eq!(
        captured.node_attr(0, &"value".into()).unwrap(),
        Some(V::Int(2))
    );
    assert!(captured.write_state_id().is_none());
    assert_eq!(graph.read_view_at(9, None).unwrap().node_count(), 0);
}
#[test]
fn detached_plan_and_recorded_cutoff_pin_accepted_history_and_derived_tables() {
    let mut graph = Graph::new_publishing();
    let first = create(&mut graph);
    let recorded = graph
        .publication_metadata(first)
        .unwrap()
        .unwrap()
        .recorded_at;
    let plan = graph.prepare_read_at(25, None).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    update(&mut graph, 20, V::Int(2));
    let old = plan.read_view().unwrap();
    let cutoff = graph.read_view_at(25, Some(recorded)).unwrap();
    for view in [old, cutoff] {
        assert_eq!(view.node_attr(0, &"value".into()).unwrap(), Some(V::Int(1)));
        let selected = view.select_nodes(&[0]).unwrap();
        assert_eq!(
            selected.read_state().read_time(),
            view.read_state().read_time()
        );
        let table = selected.nodes_table().unwrap();
        assert_eq!(table.base().source_state_id(), None);
        assert_eq!(
            table.base().source_views()[0].read_state().read_time(),
            view.read_state().read_time()
        );
    }
}
#[test]
fn branch_scope_excludes_sibling_publications() {
    let mut graph = Graph::new_publishing();
    create(&mut graph);
    graph.create_branch("sibling".into()).unwrap();
    update(&mut graph, 20, V::Int(2));
    graph.checkout_branch("sibling".into()).unwrap();
    assert_eq!(
        graph
            .read_view_at(25, None)
            .unwrap()
            .node_attr(0, &"value".into())
            .unwrap(),
        Some(V::Int(1))
    );
}
#[test]
fn topology_death_and_invalid_intervals_are_checked_at_effective_boundaries() {
    let mut graph = Graph::new_publishing();
    create(&mut graph);
    publish(
        &mut graph,
        GraphBatch {
            remove_nodes: vec![N::Existing(0)],
            effective_at: Some(20),
            ..Default::default()
        },
    );
    let before = graph.read_view_at(19, None).unwrap();
    assert_eq!((before.node_count(), before.edge_count()), (2, 1));
    let after = graph.read_view_at(20, None).unwrap();
    assert_eq!((after.node_count(), after.edge_count()), (1, 0));
    let mut invalid = Graph::new_publishing();
    create(&mut invalid);
    let head = invalid.current_state_id();
    let batch = GraphBatch {
        remove_nodes: vec![N::Existing(0)],
        remove_edges: vec![groggy::BatchEdgeRef::Existing(0)],
        effective_at: Some(20),
        change_times: vec![(T::EdgeRemoved(groggy::BatchEdgeRef::Existing(0)), 30)],
        ..Default::default()
    };
    assert!(invalid
        .apply_batch(batch, head, "invalid".into(), "test".into())
        .unwrap_err()
        .to_string()
        .contains("endpoint timeline"));
    assert_eq!(invalid.current_state_id(), head);
    assert!(!invalid.has_uncommitted_changes());
    assert_eq!(invalid.read_view_at(19, None).unwrap().edge_count(), 1);
}
#[test]
fn legacy_or_dirty_histories_reject_and_empty_graph_is_readable() {
    let mut graph = Graph::new();
    assert_eq!(graph.read_view_at(i64::MAX, None).unwrap().node_count(), 0);
    graph.add_node();
    assert!(graph
        .read_view_at(0, None)
        .unwrap_err()
        .to_string()
        .contains("clean published"));
    graph.commit("legacy".into(), "test".into()).unwrap();
    assert!(graph
        .read_view_at(i64::MAX, None)
        .unwrap_err()
        .to_string()
        .contains("precise publication"));
}

#[test]
fn mixed_property_clocks_and_acceptance_ties_match_independent_cell_selection() {
    let mut graph = Graph::new_publishing();
    publish(
        &mut graph,
        GraphBatch {
            add_nodes: 1,
            effective_at: Some(i64::MIN),
            ..Default::default()
        },
    );
    let assignments = [(20, 4), (-10, 2), (20, 7), (0, 3), (i64::MAX, 9)];
    for &(at, value) in &assignments {
        update(&mut graph, at, V::Int(value));
    }
    for at in [i64::MIN, -11, -10, 0, 19, 20, 21, i64::MAX] {
        let expected = assignments
            .iter()
            .enumerate()
            .filter(|(_, (time, _))| *time <= at)
            .max_by_key(|(accepted, (time, _))| (*time, *accepted))
            .map(|(_, (_, value))| V::Int(*value));
        assert_eq!(
            graph
                .read_view_at(at, None)
                .unwrap()
                .node_attr(0, &"value".into())
                .unwrap(),
            expected
        );
    }
}

#[test]
fn prepared_native_durable_read_outlives_store_and_replays_on_another_thread() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("timed.ggj");
    let mut store = groggy::DurableGraph::create(&path, groggy::GraphConfig::new(), None).unwrap();
    let batch = GraphBatch {
        add_nodes: 1,
        effective_at: Some(-100),
        node_attrs: HashMap::from([("value".into(), vec![(N::Added(0), V::Float64(1.25))])]),
        ..Default::default()
    };
    store
        .apply_batch(batch, 0, "initial".into(), "test".into())
        .unwrap();
    let plan = store.prepare_read_at(-100, None).unwrap();
    drop(store);
    let view = std::thread::spawn(move || plan.read_view())
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(
        view.node_attr(0, &"value".into()).unwrap(),
        Some(V::Float64(1.25))
    );
    let reopened = groggy::DurableGraph::open(&path).unwrap();
    assert_eq!(reopened.read_view_at(-101, None).unwrap().node_count(), 0);
    assert_eq!(reopened.read_view_at(-100, None).unwrap().node_count(), 1);
}
