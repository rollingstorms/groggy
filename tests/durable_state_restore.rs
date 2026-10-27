#![cfg(unix)]
use groggy::{AttrValue as V, DurableGraph, Graph, GraphBatch, GraphSchema};
#[test]
fn durable_restore_replays_reference_only_change_keeps_history_ids_and_supports_backup_collection()
{
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph");
    let backup = dir.path().join("backup");
    let mut source = Graph::new();
    source.add_node().unwrap();
    source.set_node_attr(0, "value".into(), V::Int(1)).unwrap();
    let target = source.current_state_id();
    source.set_node_attr(0, "value".into(), V::Int(2)).unwrap();
    let extra = source.add_node().unwrap();
    let future = source.current_state_id();
    let future_pin = source.read_view().unwrap();
    let versions = source.history_records().len();
    let mut store = DurableGraph::create_from_graph(&path, &source, None).unwrap();
    let outcome = store
        .graph()
        .unwrap()
        .prepare_state_restore(target, future, groggy::StateRestoreMode::Switch)
        .unwrap()
        .execute()
        .unwrap();
    assert_eq!(store.finish_state_restore(outcome).unwrap(), target);
    assert_eq!(store.graph().unwrap().history_records().len(), versions);
    assert!(store.graph().unwrap().snapshot_at_commit(future).is_ok());
    assert_eq!(
        store
            .graph()
            .unwrap()
            .get_node_attr(0, &"value".into())
            .unwrap(),
        Some(V::Int(1))
    );
    assert!(!store.graph().unwrap().has_node(extra));
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        store
            .restore_state(target, target, groggy::StateRestoreMode::Switch)
            .unwrap(),
        target
    );
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    assert!(store
        .restore_state(target, future, groggy::StateRestoreMode::Switch)
        .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    store.verify_storage().unwrap();
    store.backup(&backup).unwrap();
    drop(store);
    for path in [&path, &backup] {
        let mut reopened = DurableGraph::open(path).unwrap();
        assert_eq!(reopened.graph().unwrap().current_state_id(), target);
        assert!(reopened.graph().unwrap().snapshot_at_commit(future).is_ok());
        let added = reopened
            .apply_batch(
                GraphBatch {
                    add_nodes: 1,
                    ..Default::default()
                },
                target,
                "new path".into(),
                "test".into(),
            )
            .unwrap();
        assert_eq!(added.added_nodes, vec![extra + 1]);
        assert!(added.state_id > future);
        reopened
            .restore_state(target, added.state_id, groggy::StateRestoreMode::Switch)
            .unwrap();
        assert!(reopened.gc_history(target).unwrap() > 0);
        assert!(reopened
            .graph()
            .unwrap()
            .snapshot_at_commit(future)
            .is_err());
        reopened.checkpoint_storage(target).unwrap();
        reopened.verify_storage().unwrap();
    }
    assert!(future_pin.contains_node(extra));
}
#[test]
fn restoration_checks_current_installed_policy_and_rejects_foreign_closed_owner_outcomes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph");
    let mut source = Graph::new();
    source.add_node().unwrap();
    source.set_node_labels(0, vec!["Entity".into()]).unwrap();
    source
        .set_node_attr(0, "value".into(), V::Text("old".into()))
        .unwrap();
    let target = source.current_state_id();
    source.set_node_attr(0, "value".into(), V::Int(1)).unwrap();
    let current = source.current_state_id();
    let mut store = DurableGraph::create_from_graph(&path, &source, None).unwrap();
    let outcome = store
        .graph()
        .unwrap()
        .prepare_state_restore(target, current, groggy::StateRestoreMode::Switch)
        .unwrap()
        .execute()
        .unwrap();
    let schema =
        GraphSchema::from_json(r#"{"node_labels":{"Entity":{"value":{"type":"int"}}}}"#).unwrap();
    store.set_schema(Some(schema), current, 0).unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(store.finish_state_restore(outcome).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let outcome = store
        .graph()
        .unwrap()
        .prepare_state_restore(current, current, groggy::StateRestoreMode::Switch)
        .unwrap()
        .execute()
        .unwrap();
    drop(store);
    let mut reopened = DurableGraph::open(&path).unwrap();
    assert!(reopened.finish_state_restore(outcome).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(reopened.graph().unwrap().current_state_id(), current);
}
