use groggy::{AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch};
use std::collections::HashMap;
#[test]
fn imported_journal_retains_full_graph_then_accepts_and_recovers_new_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("import.ggj");
    let backup = dir.path().join("backup.ggj");
    let mut graph = Graph::new_publishing();
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 2,
                add_edges: vec![(N::Added(0), N::Added(1))],
                node_attrs: HashMap::from([("v".into(), vec![(N::Added(0), V::Int(1))])]),
                node_external_ids: vec![(N::Added(0), "old".into())],
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "root".into(),
            "test".into(),
        )
        .unwrap();
    let root = graph.current_state_id();
    graph.create_branch("root".into()).unwrap();
    graph
        .writer("test")
        .set_node_external_id(0, "new".into())
        .unwrap();
    let group = graph
        .retain_group(&graph.read_view().unwrap(), "source".into())
        .unwrap();
    let meta = graph.create_meta_node(group).unwrap();
    graph
        .writer("test")
        .with_effective_at(Some(20))
        .unset_node_attr(0, "v".into())
        .unwrap();
    let unused = graph.try_add_node().unwrap();
    graph.writer("test").remove_nodes(vec![unused]).unwrap();
    graph.checkpoint("label".into(), "test".into()).unwrap();
    let state = graph.current_state_id();
    let clocks = graph.publication_metadata(root).unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
    assert_eq!(graph.current_state_id(), state);
    assert_eq!(store.graph().unwrap().current_state_id(), state);
    assert!(DurableGraph::open(&path).is_err());
    let result = store
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                effective_at: Some(30),
                ..Default::default()
            },
            state,
            "after".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(result.added_nodes, vec![unused + 1]);
    let latest = result.state_id;
    store.backup(&backup).unwrap();
    drop(store);
    drop(graph);
    for path in [&path, &backup] {
        let mut store = DurableGraph::open(path).unwrap();
        let graph = store.graph().unwrap();
        assert_eq!(graph.current_state_id(), latest);
        assert_eq!(graph.publication_metadata(root).unwrap(), clocks);
        assert_eq!(graph.checkpoints().len(), 1);
        assert_eq!(
            graph.allocated_entity_for_external_id("old"),
            Some(groggy::ExternalEntityRef::Node(0))
        );
        assert_eq!(
            graph.allocated_entity_for_external_id("new"),
            Some(groggy::ExternalEntityRef::Node(0))
        );
        assert_eq!(
            graph
                .read_view()
                .unwrap()
                .meta_subgraph(meta)
                .unwrap()
                .node_attr(0, &"v".into())
                .unwrap(),
            Some(V::Int(1))
        );
        assert_eq!(graph.get_node_attr(0, &"v".into()).unwrap(), None);
        store.checkout_branch("root".into(), latest).unwrap();
        assert_eq!(
            store
                .graph()
                .unwrap()
                .get_node_attr(0, &"v".into())
                .unwrap(),
            Some(V::Int(1))
        );
    }
}
#[test]
fn rejected_imports_do_not_create_or_replace_destinations_or_accept_pending_source_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let mut source = Graph::new_manual();
    source.add_node().unwrap();
    assert!(DurableGraph::create_from_graph(&path, &source, None).is_err());
    assert!(!path.exists());
    source.commit("accept".into(), "test".into()).unwrap();
    std::fs::write(&path, b"existing").unwrap();
    assert!(DurableGraph::create_from_graph(&path, &source, None).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"existing");
    assert_eq!(source.node_count(), 1);
    assert!(!source.has_uncommitted_changes());
}

#[test]
fn imported_schema_policy_is_checked_before_creation_and_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("schema");
    let mut graph = Graph::new_publishing();
    let node = graph.try_add_node().unwrap();
    graph.set_node_labels(node, vec!["Person".into()]).unwrap();
    let schema = groggy::GraphSchema::from_json(
        r#"{"node_labels":{"Person":{"name":{"type":"text","required":true}}}}"#,
    )
    .unwrap();
    assert!(DurableGraph::create_from_graph(&path, &graph, Some(schema.clone())).is_err());
    assert!(!path.exists());
    graph
        .writer("test")
        .set_node_attr(node, "name".into(), V::Text("Ada".into()))
        .unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &graph, Some(schema)).unwrap();
    let state = store.graph().unwrap().current_state_id();
    let bytes = std::fs::metadata(&path).unwrap().len();
    assert!(store
        .apply_batch(
            GraphBatch {
                unset_node_attrs: HashMap::from([("name".into(), vec![N::Existing(node)])]),
                ..Default::default()
            },
            state,
            "bad".into(),
            "test".into()
        )
        .is_err());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes);
    drop(store);
    let mut store = DurableGraph::open(&path).unwrap();
    assert!(store
        .apply_batch(
            GraphBatch {
                node_attrs: HashMap::from([("name".into(), vec![(N::Existing(node), V::Int(4))])]),
                ..Default::default()
            },
            state,
            "bad".into(),
            "test".into()
        )
        .is_err());
}
