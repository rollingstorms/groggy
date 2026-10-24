use groggy::{AttrValue, DurableGraph, Graph, GraphConfig};

#[test]
fn checked_writes_publish_once_and_checkpoints_preserve_state_and_clocks() {
    let mut graph = Graph::new();
    let node = graph.try_add_node().unwrap();
    let first = graph.current_state_id();
    let frozen = graph.read_view().unwrap();
    let clocks = graph.publication_metadata(first).unwrap();
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(
        graph.commit("first label".into(), "test".into()).unwrap(),
        first
    );
    assert_eq!(
        graph
            .checkpoint("second label".into(), "test".into())
            .unwrap(),
        first
    );
    assert_eq!(graph.current_state_id(), first);
    assert_eq!(graph.commit_history().len(), 1);
    assert_eq!(graph.publication_metadata(first).unwrap(), clocks);
    assert_eq!(graph.checkpoints().len(), 2);
    assert_eq!(graph.checkpoints()[0].message, "first label");
    assert_eq!(graph.checkpoints()[1].branch, "main");
    assert!(frozen.contains_node(node));
    let next = graph
        .writer("test")
        .set_node_attr(node, "x".into(), AttrValue::Int(3))
        .unwrap();
    assert_eq!(next, first + 1);
    assert_eq!(graph.checkpoints()[0].state_id, first);
    assert!(graph
        .checkpoint_at(first, "stale".into(), "test".into())
        .is_err());
    assert_eq!(graph.checkpoints().len(), 2);
}

#[test]
fn empty_checkpoints_and_checked_bulk_constructors_do_not_add_phantom_versions() {
    let mut graph = Graph::new();
    assert_eq!(graph.commit("empty".into(), "test".into()).unwrap(), 0);
    assert_eq!(
        graph
            .checkpoint("empty again".into(), "test".into())
            .unwrap(),
        0
    );
    assert!(graph.try_add_nodes(0).unwrap().is_empty());
    let nodes = graph.try_add_nodes(10).unwrap();
    assert_eq!(nodes, (0..10).collect::<Vec<_>>());
    assert_eq!(graph.current_state_id(), 1);
    let mut writer = graph.writer("writer");
    assert_eq!(writer.checkpoint("bulk checkpoint").unwrap(), 1);
    assert_eq!(writer.state_id(), 1);
    assert_eq!(writer.graph().commit_history().len(), 1);
}

#[test]
fn legacy_manual_writes_must_be_published_before_an_explicit_clean_checkpoint() {
    let mut graph = Graph::new();
    let node = graph.add_node();
    assert!(graph.has_uncommitted_changes());
    assert!(graph.checkpoint("dirty".into(), "test".into()).is_err());
    assert!(graph.try_add_node().is_err());
    assert_eq!(graph.node_count(), 1);
    let state = graph
        .commit("legacy compatibility".into(), "test".into())
        .unwrap();
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(graph.checkpoints()[0].state_id, state);
    assert!(graph
        .snapshot_at_commit(state)
        .unwrap()
        .node_attr(node, &"entity_type".into())
        .is_some());
    assert_eq!(graph.try_add_node().unwrap(), 1);
}

#[test]
fn durable_checkpoints_preserve_metadata_across_branches_reopen_and_backup() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("graph.ggj");
    let backup = directory.path().join("backup.ggj");
    let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    assert_eq!(
        store.checkpoint(0, "empty".into(), "test".into()).unwrap(),
        0
    );
    let first = store
        .apply_batch(
            groggy::GraphBatch {
                add_nodes: 1,
                ..Default::default()
            },
            0,
            "node".into(),
            "test".into(),
        )
        .unwrap()
        .state_id;
    store.create_branch("other".into(), first).unwrap();
    store.checkout_branch("other".into(), first).unwrap();
    store
        .checkpoint(first, "branch label".into(), "author".into())
        .unwrap();
    let records = store.graph().unwrap().checkpoints().to_vec();
    let clocks = store.graph().unwrap().publication_metadata(first).unwrap();
    let bytes = std::fs::metadata(&path).unwrap().len();
    assert!(store
        .checkpoint(first + 1, "stale".into(), "test".into())
        .is_err());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes);
    store.backup(&backup).unwrap();
    drop(store);
    for source in [&path, &backup] {
        let recovered = DurableGraph::open(source).unwrap();
        assert_eq!(recovered.graph().unwrap().checkpoints(), records);
        assert_eq!(recovered.graph().unwrap().current_state_id(), first);
        assert_eq!(
            recovered
                .graph()
                .unwrap()
                .publication_metadata(first)
                .unwrap(),
            clocks
        );
    }
}
