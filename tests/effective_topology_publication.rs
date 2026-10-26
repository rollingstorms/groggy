use groggy::{BatchChangeRef as T, BatchEdgeRef as E, BatchNodeRef as N, Graph, GraphBatch};
fn apply(graph: &mut Graph, batch: GraphBatch) -> groggy::GraphResult<groggy::BatchResult> {
    graph.apply_batch(
        batch,
        graph.current_state_id(),
        "test".into(),
        "test".into(),
    )
}
fn nodes(at: i64) -> GraphBatch {
    GraphBatch {
        add_nodes: 2,
        effective_at: Some(at),
        ..Default::default()
    }
}
#[test]
fn edge_before_endpoint_birth_rejects_without_reserving_ids() {
    let mut graph = Graph::new_publishing();
    apply(&mut graph, nodes(10)).unwrap();
    let head = graph.current_state_id();
    assert!(apply(
        &mut graph,
        GraphBatch {
            add_edges: vec![(N::Existing(0), N::Existing(1))],
            effective_at: Some(9),
            ..Default::default()
        }
    )
    .unwrap_err()
    .to_string()
    .contains("endpoint timeline"));
    assert_eq!(graph.current_state_id(), head);
    assert_eq!(graph.read_view().unwrap().edge_count(), 0);
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(
        apply(
            &mut graph,
            GraphBatch {
                add_edges: vec![(N::Existing(0), N::Existing(1))],
                effective_at: Some(10),
                ..Default::default()
            }
        )
        .unwrap()
        .added_edges,
        vec![0]
    );
}
#[test]
fn retroactive_node_death_checks_retained_edges_even_after_their_current_removal() {
    let mut graph = Graph::new_publishing();
    apply(&mut graph, nodes(10)).unwrap();
    apply(
        &mut graph,
        GraphBatch {
            add_edges: vec![(N::Existing(0), N::Existing(1))],
            effective_at: Some(20),
            ..Default::default()
        },
    )
    .unwrap();
    apply(
        &mut graph,
        GraphBatch {
            remove_edges: vec![E::Existing(0)],
            effective_at: Some(40),
            ..Default::default()
        },
    )
    .unwrap();
    let head = graph.current_state_id();
    assert!(apply(
        &mut graph,
        GraphBatch {
            remove_nodes: vec![N::Existing(0)],
            effective_at: Some(30),
            ..Default::default()
        }
    )
    .is_err());
    assert_eq!(graph.current_state_id(), head);
    apply(
        &mut graph,
        GraphBatch {
            remove_nodes: vec![N::Existing(0)],
            effective_at: Some(40),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(graph.read_view_at(39, None).unwrap().edge_count(), 1);
    assert_eq!(graph.read_view_at(40, None).unwrap().edge_count(), 0);
}
#[test]
fn same_batch_clocks_extremes_loops_parallel_edges_and_branch_rebuilds() {
    let mut graph = Graph::new_publishing();
    let mut batch = nodes(i64::MIN);
    batch.add_edges = vec![
        (N::Added(0), N::Added(0)),
        (N::Added(0), N::Added(1)),
        (N::Added(0), N::Added(1)),
    ];
    apply(&mut graph, batch).unwrap();
    graph.create_branch("other".into()).unwrap();
    apply(
        &mut graph,
        GraphBatch {
            remove_nodes: vec![N::Existing(0)],
            effective_at: Some(i64::MAX),
            ..Default::default()
        },
    )
    .unwrap();
    graph.checkout_branch("other".into()).unwrap();
    apply(
        &mut graph,
        GraphBatch {
            add_nodes: 1,
            effective_at: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(graph.read_view_at(0, None).unwrap().edge_count(), 3);
    let mut invalid = Graph::new_publishing();
    let mut batch = nodes(20);
    batch.add_edges = vec![(N::Added(0), N::Added(1))];
    batch.change_times = vec![(T::EdgeAdded(E::Added(0)), 19)];
    assert!(apply(&mut invalid, batch).is_err());
    assert_eq!(
        apply(&mut invalid, nodes(0)).unwrap().added_nodes,
        vec![0, 1]
    );
}
#[test]
fn legacy_untimed_compatibility_is_explicitly_distinct_from_timed_guarantees() {
    let mut graph = Graph::new();
    graph.add_nodes(2);
    graph.commit("manual".into(), "test".into()).unwrap();
    assert!(apply(
        &mut graph,
        GraphBatch {
            add_edges: vec![(N::Existing(0), N::Existing(1))],
            effective_at: Some(10),
            ..Default::default()
        }
    )
    .unwrap_err()
    .to_string()
    .contains("precisely timed ancestry"));
    assert!(apply(
        &mut graph,
        GraphBatch {
            add_edges: vec![(N::Existing(0), N::Existing(1))],
            ..Default::default()
        }
    )
    .is_ok());
}
#[test]
fn property_time_override_does_not_turn_untimed_legacy_topology_into_a_timed_write() {
    let mut graph = Graph::new();
    graph.add_node();
    graph.commit("legacy".into(), "test".into()).unwrap();
    let result = apply(
        &mut graph,
        GraphBatch {
            add_nodes: 1,
            node_attrs: std::collections::HashMap::from([(
                "score".into(),
                vec![(N::Added(0), groggy::AttrValue::Int(1))],
            )]),
            change_times: vec![(T::NodeAttribute(N::Added(0), "score".into()), -10)],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.added_nodes, vec![1]);
}

#[test]
fn durable_rejection_leaves_journal_and_head_unchanged_and_reopen_uses_original_clocks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("timed.ggj");
    let mut store = groggy::DurableGraph::create(&path, groggy::GraphConfig::new(), None).unwrap();
    let first = store
        .apply_batch(nodes(i64::MAX), 0, "future".into(), "test".into())
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let edge = GraphBatch {
        add_edges: vec![(N::Existing(0), N::Existing(1))],
        effective_at: Some(i64::MAX - 1),
        ..Default::default()
    };
    assert!(store
        .apply_batch(edge, first.state_id, "bad".into(), "test".into())
        .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let edge = GraphBatch {
        add_edges: vec![(N::Existing(0), N::Existing(1))],
        effective_at: Some(i64::MAX),
        ..Default::default()
    };
    store
        .apply_batch(edge, first.state_id, "valid".into(), "test".into())
        .unwrap();
    drop(store);
    let store = groggy::DurableGraph::open(&path).unwrap();
    assert_eq!(store.read_view_at(i64::MAX, None).unwrap().edge_count(), 1);
}
