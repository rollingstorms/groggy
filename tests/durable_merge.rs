#![cfg(unix)]
use groggy::{AttrValue as V, DurableGraph, Graph, GraphBatch, GraphMergeOptions, GraphSchema};
fn source() -> (Graph, u64, u64) {
    let mut g = Graph::new();
    let base = g
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
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
    g.writer("test")
        .with_effective_at(Some(20))
        .set_node_attr(0, "left".into(), V::Int(1))
        .unwrap();
    let left = g.current_state_id();
    g.checkout_branch("right".into()).unwrap();
    g.writer("test")
        .with_effective_at(Some(30))
        .set_node_attr(0, "right".into(), V::Float64(-0.0))
        .unwrap();
    let right = g.current_state_id();
    g.checkout_branch("main".into()).unwrap();
    assert!(left > base);
    (g, left, right)
}
#[test]
fn durable_merge_preserves_two_parents_exact_values_reopen_backup_maintenance_and_future_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph");
    let backup = dir.path().join("backup");
    let (source, left, right) = source();
    let mut store = DurableGraph::create_from_graph(&path, &source, None).unwrap();
    let pin = store.graph().unwrap().read_view().unwrap();
    let outcome = store
        .graph()
        .unwrap()
        .prepare_merge_read(right, None, left)
        .unwrap()
        .execute()
        .unwrap()
        .into_outcome()
        .unwrap();
    let id = store
        .finish_merge(
            outcome,
            GraphMergeOptions {
                effective_at: Some(40),
                ..Default::default()
            },
            "merge".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(id, right + 1);
    assert!(pin.node_attr(0, &"right".into()).unwrap().is_none());
    store.verify_storage().unwrap();
    store.backup(&backup).unwrap();
    drop(store);
    for path in [&path, &backup] {
        let mut store = DurableGraph::open(path).unwrap();
        let graph = store.graph().unwrap();
        assert_eq!(graph.current_state_id(), id);
        let record = graph
            .history_records()
            .into_iter()
            .find(|c| c.id == id)
            .unwrap();
        assert_eq!(record.parents, vec![left, right]);
        assert_eq!(record.publication.as_ref().unwrap().effective_at, 40);
        if let Some(V::Float64(v)) = graph.get_node_attr(0, &"right".into()).unwrap() {
            assert_eq!(v.to_bits(), (-0.0_f64).to_bits());
        } else {
            panic!()
        }
        assert!(graph
            .read_view_at(39, None)
            .unwrap()
            .node_attr(0, &"right".into())
            .unwrap()
            .is_none());
        assert!(graph
            .read_view_at(40, None)
            .unwrap()
            .node_attr(0, &"right".into())
            .unwrap()
            .is_some());
        store.prune_branch("right".into(), id).unwrap();
        assert!(store.graph().unwrap().snapshot_at_commit(right).is_ok());
        store.checkpoint_storage(id).unwrap();
        store.compact_storage_with_reclamation(id, true).unwrap();
        store.verify_storage().unwrap();
        assert_eq!(
            store
                .apply_batch(
                    GraphBatch {
                        add_nodes: 1,
                        ..Default::default()
                    },
                    id,
                    "next".into(),
                    "test".into()
                )
                .unwrap()
                .added_nodes,
            vec![1]
        );
    }
}
#[test]
fn durable_merge_enforces_latest_installed_and_additional_schema_before_bytes_or_state_change() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph");
    let (mut source, left, right) = source();
    source.set_node_labels(0, vec!["Entity".into()]).unwrap();
    let current = source.current_state_id();
    let mut store = DurableGraph::create_from_graph(&path, &source, None).unwrap();
    let outcome = store
        .graph()
        .unwrap()
        .prepare_merge_read(right, None, current)
        .unwrap()
        .execute()
        .unwrap()
        .into_outcome()
        .unwrap();
    let schema = GraphSchema::from_json(
        r#"{"node_labels":{"Entity":{"right":{"type":"text","required":false}}}}"#,
    )
    .unwrap();
    store.set_schema(Some(schema), current, 0).unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(store
        .finish_merge(
            outcome,
            GraphMergeOptions::default(),
            "bad".into(),
            "test".into()
        )
        .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(store.graph().unwrap().current_state_id(), current);
    assert!(current > left);
}

#[test]
fn additional_merge_schema_can_reject_without_installing_a_policy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph");
    let (mut graph, _, right) = source();
    graph.set_node_labels(0, vec!["Entity".into()]).unwrap();
    let left = graph.current_state_id();
    let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
    let before = std::fs::read(&path).unwrap();
    let outcome = store
        .graph()
        .unwrap()
        .prepare_merge_read(right, None, left)
        .unwrap()
        .execute()
        .unwrap()
        .into_outcome()
        .unwrap();
    let schema = GraphSchema::from_json(
        r#"{"node_labels":{"Entity":{"right":{"type":"text","required":false}}}}"#,
    )
    .unwrap();
    assert!(store
        .finish_merge_with_schema(
            outcome,
            GraphMergeOptions::default(),
            "bad".into(),
            "test".into(),
            Some(&schema)
        )
        .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(store.schema_revision(), 0);
}
