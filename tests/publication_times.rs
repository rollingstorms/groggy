use groggy::{
    AttrValue, BatchChangeRef as T, BatchNodeRef as N, ChangeKey as K, Graph, GraphBatch,
};
use std::collections::HashMap;
#[test]
fn batch_default_and_individual_times_are_retained_with_one_recorded_clock() {
    let mut graph = Graph::new();
    let batch = GraphBatch {
        add_nodes: 1,
        effective_at: Some(100),
        node_attrs: HashMap::from([
            (
                "name".into(),
                vec![(N::Added(0), AttrValue::Text("Ada".into()))],
            ),
            ("age".into(), vec![(N::Added(0), AttrValue::Int(42))]),
        ]),
        change_times: vec![
            (T::NodeAdded(N::Added(0)), 50),
            (T::NodeAttribute(N::Added(0), "age".into()), -10),
        ],
        ..Default::default()
    };
    let result = graph
        .apply_batch(batch, 0, "timed".into(), "test".into())
        .unwrap();
    let metadata = graph
        .publication_metadata(result.state_id)
        .unwrap()
        .unwrap();
    assert_eq!(metadata.effective_at, 100);
    assert!(metadata.recorded_at > 100);
    assert_eq!(metadata.change_times[&K::NodeAdded(0)], 50);
    assert_eq!(
        metadata.change_times[&K::NodeAttribute(0, "entity_type".into())],
        50
    );
    assert_eq!(
        metadata.change_times[&K::NodeAttribute(0, "name".into())],
        50
    );
    assert_eq!(
        metadata.change_times[&K::NodeAttribute(0, "age".into())],
        -10
    );
    assert_eq!(
        graph.commit_history()[0].timestamp,
        metadata.recorded_at as u64 / 1_000_000
    );
}
#[test]
fn automatic_default_is_one_clock_and_legacy_manual_commits_are_distinct() {
    let mut graph = Graph::new();
    let id = graph.writer("test").add_node(HashMap::new()).unwrap();
    let state = graph.current_state_id();
    let metadata = graph.publication_metadata(state).unwrap().unwrap();
    assert_eq!(metadata.recorded_at, metadata.effective_at);
    assert!(metadata
        .change_times
        .values()
        .all(|at| *at == metadata.recorded_at));
    graph
        .set_node_attr(id, "legacy".into(), AttrValue::Bool(true))
        .unwrap();
    let legacy = graph.commit("manual".into(), "test".into()).unwrap();
    assert!(graph.publication_metadata(legacy).unwrap().is_none());
    assert_eq!(
        graph.publication_metadata(state).unwrap().unwrap(),
        metadata
    );
}
#[test]
fn implicit_edge_death_uses_earliest_removed_endpoint_time() {
    let mut graph = Graph::new();
    let ids = graph.writer("test").add_nodes(2).unwrap();
    let edge = graph
        .writer("test")
        .add_edge(ids[0], ids[1], HashMap::new())
        .unwrap();
    let state = graph.current_state_id();
    let batch = GraphBatch {
        remove_nodes: vec![N::Existing(ids[0]), N::Existing(ids[1])],
        change_times: vec![
            (T::NodeRemoved(N::Existing(ids[0])), 20),
            (T::NodeRemoved(N::Existing(ids[1])), 10),
        ],
        ..Default::default()
    };
    let result = graph
        .apply_batch(batch, state, "removed".into(), "test".into())
        .unwrap();
    let metadata = graph
        .publication_metadata(result.state_id)
        .unwrap()
        .unwrap();
    assert_eq!(metadata.change_times[&K::EdgeRemoved(edge)], 10);
}
#[test]
fn invalid_or_duplicate_time_targets_reject_before_allocation() {
    for times in [
        vec![(T::NodeAttribute(N::Added(0), "missing".into()), 10)],
        vec![
            (T::NodeAdded(N::Added(0)), 10),
            (T::NodeAdded(N::Added(0)), 20),
        ],
    ] {
        let mut graph = Graph::new();
        let batch = GraphBatch {
            add_nodes: 1,
            change_times: times,
            ..Default::default()
        };
        assert!(graph
            .apply_batch(batch, 0, "invalid".into(), "test".into())
            .unwrap_err()
            .to_string()
            .contains("change_times"));
        assert_eq!(graph.current_state_id(), 0);
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(graph.add_node(), 0);
    }
}
