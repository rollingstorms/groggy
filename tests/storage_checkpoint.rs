#![cfg(unix)]
use groggy::{AttrValue as V, BatchNodeRef as N, DurableGraph, GraphBatch, GraphConfig};
use std::collections::HashMap;
#[test]
fn storage_checkpoint_preserves_schema_history_branches_pool_and_pins_then_continues() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let backup = dir.path().join("backup");
    let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    store
        .apply_batch(
            GraphBatch {
                add_nodes: 2,
                add_edges: vec![(N::Added(0), N::Added(1))],
                node_external_ids: vec![(N::Added(0), "old".into())],
                node_attrs: HashMap::from([("v".into(), vec![(N::Added(0), V::Int(1))])]),
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "root".into(),
            "test".into(),
        )
        .unwrap();
    let root = store.graph().unwrap().current_state_id();
    store.create_branch("root".into(), root).unwrap();
    store
        .set_schema(Some(groggy::GraphSchema::default()), root, 0)
        .unwrap();
    store.set_schema(None, root, 1).unwrap();
    let group = store
        .retain_group(
            &store.graph().unwrap().read_view().unwrap(),
            "source".into(),
        )
        .unwrap();
    let meta = store.create_meta_node(group, root, Some(15)).unwrap();
    let state = store.graph().unwrap().current_state_id();
    store
        .apply_batch(
            GraphBatch {
                node_external_ids: vec![(N::Existing(0), "new".into())],
                unset_node_attrs: HashMap::from([("v".into(), vec![N::Existing(0)])]),
                effective_at: Some(20),
                ..Default::default()
            },
            state,
            "update".into(),
            "test".into(),
        )
        .unwrap();
    let state = store.graph().unwrap().current_state_id();
    store
        .checkpoint(state, "label".into(), "test".into())
        .unwrap();
    let pin = store.graph().unwrap().read_view().unwrap();
    let versions = store.graph().unwrap().history_records().len();
    let clocks = store.graph().unwrap().publication_metadata(root).unwrap();
    let values = store
        .graph()
        .unwrap()
        .pool()
        .get_node_attribute_column(&"v".into())
        .unwrap()
        .len();
    let report = store.checkpoint_storage(state).unwrap();
    assert_eq!(report.selected_state, state);
    assert_eq!(report.retained_versions, versions);
    assert_eq!(
        report.current_bytes,
        std::fs::metadata(&path).unwrap().len()
    );
    assert_eq!(store.schema_revision(), 2);
    assert!(DurableGraph::open(&path).is_err());
    assert_eq!(
        pin.meta_subgraph(meta)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Int(1))
    );
    assert_eq!(pin.node_attr(0, &"v".into()).unwrap(), None);
    let after = store
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                ..Default::default()
            },
            state,
            "after".into(),
            "test".into(),
        )
        .unwrap();
    store.backup(&backup).unwrap();
    drop(store);
    for path in [&path, &backup] {
        let mut store = DurableGraph::open(path).unwrap();
        let graph = store.graph().unwrap();
        assert_eq!(graph.current_state_id(), after.state_id);
        assert_eq!(graph.publication_metadata(root).unwrap(), clocks);
        assert_eq!(graph.history_records().len(), versions + 1);
        assert_eq!(graph.checkpoints().len(), 1);
        assert_eq!(
            graph
                .pool()
                .get_node_attribute_column(&"v".into())
                .unwrap()
                .len(),
            values
        );
        assert_eq!(
            graph.allocated_entity_for_external_id("old"),
            Some(groggy::ExternalEntityRef::Node(0))
        );
        assert_eq!(store.schema_revision(), 2);
        store
            .checkout_branch("root".into(), after.state_id)
            .unwrap();
        assert_eq!(store.graph().unwrap().current_state_id(), root);
    }
}
#[test]
fn post_checkpoint_partial_tail_recovers_and_explicit_rewrites_are_repeatable() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    assert_eq!(store.checkpoint_storage(0).unwrap().retained_versions, 0);
    let report = store.checkpoint_storage(0).unwrap();
    assert_eq!(report.previous_records, 1);
    drop(store);
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    file.write_all(b"GFRAME01").unwrap();
    file.sync_all().unwrap();
    drop(file);
    let mut store = DurableGraph::open(&path).unwrap();
    assert_eq!(store.recovered_tail_bytes(), 8);
    store.checkpoint_storage(0).unwrap();
    assert_eq!(store.recovered_tail_bytes(), 0);
}
