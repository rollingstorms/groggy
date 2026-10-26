#![cfg(unix)]
use groggy::{AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch};

#[test]
fn durable_compaction_preserves_branches_clocks_groups_reservations_and_continued_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let backup = dir.path().join("backup");
    let mut source = Graph::new();
    source.add_nodes(3).unwrap();
    source.add_edge(0, 1).unwrap();
    source
        .writer("test")
        .set_node_external_id(0, "reserved".into())
        .unwrap();
    source.create_branch("base".into()).unwrap();
    for _ in 0..20 {
        source
            .set_node_attr(0, "text".into(), V::Text("raw payload".repeat(200)))
            .unwrap();
        source
            .set_edge_attr(0, "weight".into(), V::Float64(-0.0))
            .unwrap();
    }
    let group = source
        .retain_group(&source.read_view().unwrap(), "frozen".into())
        .unwrap();
    let meta = source.create_meta_node(group).unwrap();
    source.remove_node(2).unwrap();
    let state = source.current_state_id();
    let clocks = source.publication_metadata(state).unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &source, None).unwrap();
    store.set_schema(None, state, 0).unwrap();
    store
        .checkpoint(state, "label".into(), "test".into())
        .unwrap();
    let pin = store.graph().unwrap().read_view().unwrap();
    let before = store.check_integrity().unwrap();
    let original = std::fs::read(&path).unwrap();
    assert!(store.compact_storage(state - 1).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let plan = store.prepare_storage_compaction(state).unwrap();
    assert!(store.graph().is_err());
    let outcome = std::thread::spawn(move || plan.execute()).join().unwrap();
    assert!(DurableGraph::open(&path).is_err());
    let report = store.finish_storage_checkpoint(outcome).unwrap();
    let slots = report.pool_compaction.unwrap();
    assert!(slots.current_node_values < slots.previous_node_values);
    assert!(slots.current_edge_values < slots.previous_edge_values);
    assert!(report.current_bytes < report.previous_bytes);
    assert_eq!(
        report.current_bytes,
        std::fs::metadata(&path).unwrap().len()
    );
    let after = store.check_integrity().unwrap();
    assert_eq!(after.retained_versions, before.retained_versions);
    assert_eq!(after.allocated_nodes, before.allocated_nodes);
    assert_eq!(after.allocated_edges, before.allocated_edges);
    assert_eq!(
        after.retained_group_references,
        before.retained_group_references
    );
    assert_eq!(
        after.external_id_reservations,
        before.external_id_reservations
    );
    assert_eq!(after.schema_revision, before.schema_revision);
    assert_eq!(
        pin.meta_subgraph(meta)
            .unwrap()
            .node_attr(0, &"text".into())
            .unwrap(),
        Some(V::Text("raw payload".repeat(200)))
    );
    assert_eq!(
        store.graph().unwrap().publication_metadata(state).unwrap(),
        clocks
    );
    let repeat = store
        .compact_storage(state)
        .unwrap()
        .pool_compaction
        .unwrap();
    assert_eq!(repeat.previous_node_values, repeat.current_node_values);
    assert_eq!(repeat.previous_edge_values, repeat.current_edge_values);
    assert!(store
        .checkpoint_storage(state)
        .unwrap()
        .pool_compaction
        .is_none());
    let result = store
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                node_attrs: std::collections::HashMap::from([(
                    "text".into(),
                    vec![(N::Added(0), V::Text("after".into()))],
                )]),
                ..Default::default()
            },
            state,
            "after".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(result.added_nodes, vec![meta + 1]);
    let latest = store.graph().unwrap().current_state_id();
    store.backup(&backup).unwrap();
    drop(store);
    for path in [&path, &backup] {
        let mut recovered = DurableGraph::open(path).unwrap();
        assert_eq!(recovered.graph().unwrap().current_state_id(), latest);
        assert_eq!(
            recovered.check_integrity().unwrap().retained_versions,
            before.retained_versions + 1
        );
        recovered.checkout_branch("base".into(), latest).unwrap();
        assert!(recovered.graph().unwrap().has_node(2));
        assert!(pin.meta_subgraph(meta).unwrap().nodes().contains(&2));
        recovered.check_integrity().unwrap();
    }
}
