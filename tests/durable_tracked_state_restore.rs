#![cfg(unix)]
use groggy::{
    AttrValue as V, DurableGraph, Graph, GraphMergeOptions, GraphSchema, StateRestoreMode as M,
};
#[test]
fn tracked_sibling_target_is_not_an_implicit_history_pin_and_restored_payload_survives_reclamation()
{
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph");
    let backup = dir.path().join("backup");
    let mut g = Graph::new();
    g.add_node().unwrap();
    let root = g.current_state_id();
    g.create_branch("source".into()).unwrap();
    g.checkout_branch("source".into()).unwrap();
    g.set_node_attr(0, "value".into(), V::Float64(-0.0))
        .unwrap();
    g.writer("test")
        .set_node_external_id(0, "source-name".into())
        .unwrap();
    let view = g.read_view().unwrap();
    let group = g.retain_group(&view, "frozen".into()).unwrap();
    let meta = g.create_meta_node(group).unwrap();
    let target = g.current_state_id();
    g.checkout_branch("main".into()).unwrap();
    let pin = g.read_view().unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &g, None).unwrap();
    let state = store.restore_state(target, root, M::Tracked).unwrap();
    assert!(state > target);
    assert!(!pin.contains_node(meta));
    assert_eq!(
        store
            .graph()
            .unwrap()
            .node_by_external_id("source-name", None),
        Some(0)
    );
    assert!(store.prune_branch("source".into(), state).unwrap() > 0);
    assert!(store.graph().unwrap().snapshot_at_commit(target).is_err());
    store.verify_storage().unwrap();
    store.checkpoint_storage(state).unwrap();
    store.purge_unreferenced(state).unwrap();
    store.compact_storage_with_reclamation(state, true).unwrap();
    store.backup(&backup).unwrap();
    drop(store);
    for path in [&path, &backup] {
        let mut store = DurableGraph::open(path).unwrap();
        let g = store.graph().unwrap();
        assert_eq!(g.current_state_id(), state);
        assert!(g.snapshot_at_commit(target).is_err());
        let value = g
            .read_view()
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .node_attr(0, &"value".into())
            .unwrap()
            .unwrap();
        let V::Float64(value) = value else {
            panic!("not exact f64")
        };
        assert_eq!(value.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(g.node_by_external_id("source-name", None), Some(0));
        store.verify_storage().unwrap();
    }
}
#[test]
fn tracked_restoration_cannot_weaken_current_installed_or_additional_schema_and_native_writer_keeps_author(
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph");
    let mut g = Graph::new();
    g.add_node().unwrap();
    g.set_node_labels(0, vec!["Entity".into()]).unwrap();
    g.set_node_attr(0, "value".into(), V::Text("old".into()))
        .unwrap();
    let old = g.current_state_id();
    g.set_node_attr(0, "value".into(), V::Int(2)).unwrap();
    let current = g.current_state_id();
    let policy =
        GraphSchema::from_json(r#"{"node_labels":{"Entity":{"value":{"type":"int"}}}}"#).unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &g, None).unwrap();
    let outcome = store
        .graph()
        .unwrap()
        .prepare_state_restore(old, current, M::Tracked)
        .unwrap()
        .execute()
        .unwrap();
    store.set_schema(Some(policy), current, 0).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert!(store
        .finish_state_restore_with_options(
            outcome,
            GraphMergeOptions::default(),
            "bad".into(),
            "test".into(),
            Some(&GraphSchema::default())
        )
        .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    assert_eq!(store.graph().unwrap().current_state_id(), current);
    // Native writer uses its author and explicit default time for tracked mode.
    let state = g
        .writer("owner")
        .with_effective_at(Some(i64::MAX / 2))
        .restore_state(old, M::Tracked)
        .unwrap();
    let record = g
        .history_records()
        .into_iter()
        .find(|c| c.id == state)
        .unwrap();
    assert_eq!(record.author, "owner");
    assert_eq!(
        record.publication.as_ref().unwrap().effective_at,
        i64::MAX / 2
    );
}
