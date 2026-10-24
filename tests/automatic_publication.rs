use groggy::{AttrValue as V, Graph, GraphBatch, NodeUpsert};
use std::collections::{HashMap, HashSet};

fn publishing() -> Graph {
    let mut graph = Graph::new();
    graph.set_automatic_publication(true).unwrap();
    graph
}
#[test]
fn ordinary_bulk_creation_setters_and_cascade_removal_each_publish_once() {
    let mut graph = publishing();
    let nodes = graph.insert_nodes(3).unwrap();
    assert_eq!(graph.current_state_id(), 1);
    let edges = graph
        .add_edges_with_attrs(vec![
            (nodes[0], nodes[1], HashMap::from([("w".into(), V::Int(1))])),
            (nodes[1], nodes[2], HashMap::from([("w".into(), V::Int(2))])),
        ])
        .unwrap();
    assert_eq!(graph.current_state_id(), 2);
    let old = graph.read_view().unwrap();
    graph
        .set_node_attrs(HashMap::from([(
            "score".into(),
            nodes.iter().map(|&id| (id, V::Float64(0.5))).collect(),
        )]))
        .unwrap();
    assert_eq!(graph.current_state_id(), 3);
    graph
        .set_node_attr_column(
            "score".into(),
            vec![(nodes[0], V::Float64(1.5)), (nodes[2], V::Float64(2.5))],
        )
        .unwrap();
    assert_eq!(graph.current_state_id(), 4);
    graph.remove_nodes(&[nodes[1]]).unwrap();
    assert_eq!(graph.current_state_id(), 5);
    assert!(edges.iter().all(|&id| !graph.has_edge(id)));
    assert!(old.contains_node(nodes[1]));
    assert!(!graph.has_uncommitted_changes());
    assert!(graph.automatic_publication());
    let metadata = graph.publication_metadata(5).unwrap().unwrap();
    assert_eq!(metadata.change_times.len(), 3);
    assert!(metadata
        .change_times
        .values()
        .all(|&at| at == metadata.recorded_at));
    assert_eq!(graph.commit("checkpoint".into(), "test".into()).unwrap(), 5);
    assert_eq!(graph.commit_history().len(), 5);
}
#[test]
fn validation_failure_preserves_head_ids_properties_and_publication_policy() {
    let mut graph = publishing();
    let nodes = graph.insert_nodes(2).unwrap();
    let state = graph.current_state_id();
    assert!(graph
        .add_edges_with_attrs(vec![
            (nodes[0], nodes[1], HashMap::new()),
            (nodes[0], 99, HashMap::new())
        ])
        .is_err());
    assert!(graph
        .add_nodes_with_attrs(vec![
            HashMap::new(),
            HashMap::from([("entity_type".into(), V::Text("meta".into()))])
        ])
        .is_err());
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.current_state_id(), state);
    assert!(graph.automatic_publication());
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(graph.add_edges(&[(0, 1)]).unwrap(), vec![0]);
    assert_eq!(graph.insert_nodes(1).unwrap(), vec![2]);
    let state = graph.current_state_id();
    graph.insert_nodes(0).unwrap();
    graph.add_edges(&[]).unwrap();
    graph.set_node_attrs(HashMap::new()).unwrap();
    graph.remove_nodes(&[]).unwrap();
    assert_eq!(graph.current_state_id(), state);
}
#[test]
fn identity_upserts_and_explicit_batches_share_one_publication_boundary() {
    let mut graph = publishing();
    let rows = vec![
        NodeUpsert {
            external_id: "a".into(),
            attrs: HashMap::new(),
        },
        NodeUpsert {
            external_id: "b".into(),
            attrs: HashMap::new(),
        },
    ];
    assert_eq!(graph.upsert_nodes(rows.clone()).unwrap(), vec![0, 1]);
    assert_eq!(graph.current_state_id(), 1);
    graph.upsert_nodes(rows).unwrap();
    assert_eq!(graph.current_state_id(), 1);
    graph.set_node_external_id(0, "renamed".into()).unwrap();
    assert_eq!(graph.current_state_id(), 2);
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 2,
                ..Default::default()
            },
            2,
            "mixed".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(graph.current_state_id(), 3);
    assert!(graph.automatic_publication());
    graph.set_node_attr(0, "value".into(), V::Int(1)).unwrap();
    assert_eq!(graph.current_state_id(), 4);
}
#[test]
fn meta_creation_publishes_valid_system_properties_together_and_copy_is_preflighted() {
    let mut graph = publishing();
    graph.insert_nodes(2).unwrap();
    let before = graph.current_state_id();
    assert!(graph.create_meta_node(999).is_err());
    assert_eq!(graph.current_state_id(), before);
    let reference = graph
        .pool_mut()
        .store_subgraph(HashSet::from([0, 1]), HashSet::new(), "subgraph".into())
        .unwrap();
    let meta = graph.create_meta_node(reference).unwrap();
    assert!(graph.is_meta_node(meta));
    assert_eq!(graph.current_state_id(), before + 1);
    let snapshot = graph.snapshot_at_commit(before + 1).unwrap();
    assert_eq!(
        snapshot.node_attr(meta, &"entity_type".into()),
        Some(V::Text("meta".into()))
    );
    assert_eq!(
        snapshot.node_attr(meta, &"contains_subgraph".into()),
        Some(V::SubgraphRef(reference))
    );
    assert!(!graph.has_uncommitted_changes());
    let mut target = publishing();
    assert!(target.add_graph(&graph).is_err());
    assert_eq!(target.node_count(), 0);
    assert_eq!(target.current_state_id(), 0);
    let mut source = publishing();
    source.insert_nodes(2).unwrap();
    source.add_edge(0, 1).unwrap();
    source.set_node_attr(0, "x".into(), V::Int(4)).unwrap();
    target.add_graph(&source).unwrap();
    assert_eq!(target.current_state_id(), 1);
    assert_eq!(target.edge_count(), 1);
    assert_eq!(
        target.get_node_attr(0, &"x".into()).unwrap(),
        Some(V::Int(4))
    );
}
#[test]
fn explicit_legacy_writes_cannot_be_silently_mixed_into_a_publishing_graph() {
    let mut graph = publishing();
    graph.add_node(); // Existing infallible Rust compatibility API remains manual.
    assert!(graph.set_node_attr(0, "x".into(), V::Int(1)).is_err());
    assert!(graph.set_automatic_publication(false).is_err());
    graph
        .commit("legacy boundary".into(), "test".into())
        .unwrap();
    graph.set_node_attr(0, "x".into(), V::Int(1)).unwrap();
    assert!(!graph.has_uncommitted_changes());
}
