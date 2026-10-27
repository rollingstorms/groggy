#![cfg(unix)]
use groggy::{AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch};
#[test]
fn durable_endpoint_revisions_replay_checkpoint_compact_purge_backup_and_continue() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph");
    let backup = dir.path().join("backup");
    let mut source = Graph::new();
    source
        .apply_batch(
            GraphBatch {
                add_nodes: 3,
                add_edges: vec![(N::Added(0), N::Added(1))],
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "initial".into(),
            "test".into(),
        )
        .unwrap();
    source
        .writer("test")
        .with_effective_at(Some(10))
        .set_edge_attr(0, "value".into(), V::Int(9))
        .unwrap();
    source
        .writer("test")
        .with_effective_at(Some(10))
        .set_edge_external_id(0, "edge".into())
        .unwrap();
    let initial = source.current_state_id();
    let pin = source.read_view().unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &source, None).unwrap();
    let batch = GraphBatch {
        edge_endpoints: vec![(0, N::Existing(2), N::Existing(1))],
        effective_at: Some(20),
        ..Default::default()
    };
    let before = std::fs::read(&path).unwrap();
    assert!(store
        .apply_batch(batch.clone(), initial - 1, "stale".into(), "test".into())
        .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let after = store
        .apply_batch(batch, initial, "revision".into(), "test".into())
        .unwrap()
        .state_id;
    let metadata = store.graph().unwrap().publication_metadata(after).unwrap();
    for maintenance in 0..3 {
        match maintenance {
            0 => {
                store.checkpoint_storage(after).unwrap();
            }
            1 => {
                store.compact_storage_with_reclamation(after, true).unwrap();
            }
            _ => {
                store.purge_unreferenced(after).unwrap();
            }
        }
        assert_eq!(store.graph().unwrap().edge_endpoints(0).unwrap(), (2, 1));
        assert_eq!(
            store.graph().unwrap().publication_metadata(after).unwrap(),
            metadata
        );
        store.verify_storage().unwrap();
    }
    store.backup(&backup).unwrap();
    drop(store);
    for candidate in [&path, &backup] {
        let mut reopened = DurableGraph::open(candidate).unwrap();
        let graph = reopened.graph().unwrap();
        assert_eq!(graph.edge_by_external_id("edge", None), Some(0));
        assert_eq!(
            graph.get_edge_attr(0, &"value".into()).unwrap(),
            Some(V::Int(9))
        );
        assert_eq!(
            graph
                .read_view_at(19, None)
                .unwrap()
                .edge_endpoints(0)
                .unwrap(),
            (0, 1)
        );
        assert_eq!(
            graph
                .read_view_at(20, None)
                .unwrap()
                .edge_endpoints(0)
                .unwrap(),
            (2, 1)
        );
        let mut reversal = graph.edge_reversal_batch(&[0]).unwrap();
        reversal.effective_at = Some(30);
        reopened
            .apply_batch(reversal, after, "reverse".into(), "test".into())
            .unwrap();
        assert_eq!(reopened.graph().unwrap().edge_endpoints(0).unwrap(), (1, 2));
        reopened.verify_storage().unwrap();
    }
    assert_eq!(pin.edge_endpoints(0).unwrap(), (0, 1));
}
