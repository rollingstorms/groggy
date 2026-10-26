#![cfg(unix)]
use groggy::{AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch};
fn source() -> Graph {
    let mut g = Graph::new();
    g.add_node().unwrap();
    g.create_branch("discard".into()).unwrap();
    g.checkout_branch("discard".into()).unwrap();
    g.add_node().unwrap();
    g.add_edge(0, 1).unwrap();
    g.writer("test")
        .set_node_external_id(1, "reuse".into())
        .unwrap();
    g.set_node_attr(1, "payload".into(), V::Text("frozen".into()))
        .unwrap();
    let group = g
        .retain_group(&g.read_view().unwrap(), "source".into())
        .unwrap();
    g.checkout_branch("main".into()).unwrap();
    g.create_meta_node(group).unwrap();
    g
}
#[test]
fn durable_purge_installs_identical_memory_and_storage_and_keeps_schema_clocks_pins_and_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let backup = dir.path().join("backup");
    let graph = source();
    let state = graph.current_state_id();
    let clocks = graph.publication_metadata(state).unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
    store
        .set_schema(Some(groggy::GraphSchema::default()), state, 0)
        .unwrap();
    assert!(store.prune_branch("discard".into(), state).unwrap() > 0);
    store
        .checkpoint(state, "label".into(), "test".into())
        .unwrap();
    let pin = store.graph().unwrap().read_view().unwrap();
    let preview = store.graph().unwrap().preview_purge().unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(store.purge_unreferenced(state - 1).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let plan = store.prepare_storage_purge(state).unwrap();
    assert!(store.graph().is_err());
    let outcome = std::thread::spawn(move || plan.execute()).join().unwrap();
    assert!(DurableGraph::open(&path).is_err());
    let report = store.finish_storage_checkpoint(outcome).unwrap();
    assert_eq!(report.purge.unwrap(), preview);
    assert_eq!(store.graph().unwrap().current_state_id(), state);
    assert_eq!(
        store.graph().unwrap().publication_metadata(state).unwrap(),
        clocks
    );
    assert_eq!(store.schema_revision(), 1);
    assert!(!store.graph().unwrap().is_node_allocated(1));
    assert!(!store.graph().unwrap().is_edge_allocated(0));
    assert_eq!(
        store
            .graph()
            .unwrap()
            .read_view()
            .unwrap()
            .meta_subgraph(2)
            .unwrap()
            .node_attr(1, &"payload".into())
            .unwrap(),
        Some(V::Text("frozen".into()))
    );
    assert_eq!(
        pin.meta_subgraph(2)
            .unwrap()
            .node_attr(1, &"payload".into())
            .unwrap(),
        Some(V::Text("frozen".into()))
    );
    assert_eq!(store.check_integrity().unwrap().purged_nodes, 1);
    assert_eq!(store.verify_storage().unwrap().graph.purged_nodes, 1);
    let batch = GraphBatch {
        add_nodes: 1,
        add_edges: vec![(N::Existing(0), N::Added(0))],
        node_external_ids: vec![(N::Added(0), "reuse".into())],
        ..Default::default()
    };
    let accepted = store
        .apply_batch(batch, state, "new identity".into(), "test".into())
        .unwrap();
    assert_eq!(accepted.added_nodes, vec![3]);
    assert_eq!(accepted.added_edges, vec![1]);
    let latest = store.graph().unwrap().current_state_id();
    store.backup(&backup).unwrap();
    drop(store);
    for path in [&path, &backup] {
        let mut reopened = DurableGraph::open(path).unwrap();
        assert_eq!(reopened.graph().unwrap().current_state_id(), latest);
        assert!(!reopened.graph().unwrap().is_node_allocated(1));
        assert_eq!(
            reopened.graph().unwrap().node_by_external_id("reuse", None),
            Some(3)
        );
        assert_eq!(
            reopened.graph().unwrap().pool().get_edge_endpoints(1),
            Some((0, 3))
        );
        assert_eq!(reopened.graph().unwrap().checkpoints().len(), 1);
        assert_eq!(
            reopened
                .purge_unreferenced(latest)
                .unwrap()
                .purge
                .unwrap()
                .purged_nodes,
            0
        );
        reopened.verify_storage().unwrap();
    }
}
