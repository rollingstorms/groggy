use groggy::state::state::GraphSnapshot;
use groggy::{AttrValue as V, BatchNodeRef as N, Graph, GraphBatch};
use std::collections::HashMap;
fn snapshot() -> GraphSnapshot {
    let mut snapshot = GraphSnapshot::empty_with_type(42, groggy::types::GraphType::Directed);
    snapshot.active_nodes = vec![4, 7];
    snapshot.edges.insert(9, (4, 7));
    snapshot.node_attributes.insert(
        4,
        HashMap::from([
            ("v".into(), V::Null),
            ("untouched".into(), V::Bytes(vec![0, 255])),
        ]),
    );
    snapshot.node_attributes.insert(
        7,
        HashMap::from([("refs".into(), V::NodeArray(vec![4, 7]))]),
    );
    snapshot
        .edge_attributes
        .insert(9, HashMap::from([("w".into(), V::Float64(1.0000000001))]));
    snapshot
}
#[test]
fn materialized_base_requires_acceptance_and_first_commit_includes_complete_graph() {
    let mut graph = Graph::from_snapshot(snapshot()).unwrap();
    assert!(graph.has_uncommitted_changes());
    let result = graph.apply_batch(
        GraphBatch {
            add_nodes: 1,
            ..Default::default()
        },
        0,
        "bad".into(),
        "test".into(),
    );
    assert!(result.unwrap_err().to_string().contains("uncommitted"));
    graph.set_node_attr(4, "v".into(), V::Int(9)).unwrap();
    let state = graph.commit("adopt".into(), "test".into()).unwrap();
    let retained = graph.snapshot_at_commit(state).unwrap();
    assert!(retained.node_exists(4) && retained.node_exists(7) && retained.edge_exists(9));
    assert_eq!(
        retained.node_attr(4, &"untouched".into()),
        Some(V::Bytes(vec![0, 255]))
    );
    assert_eq!(retained.node_attr(4, &"v".into()), Some(V::Int(9)));
    assert_eq!(
        retained.edge_attr(9, &"w".into()),
        Some(V::Float64(1.0000000001))
    );
    assert_eq!(graph.try_add_node().unwrap(), 8);
}
#[test]
fn explicit_publishing_adoption_accepts_one_complete_typed_import_before_future_mutations() {
    let mut graph = Graph::from_snapshot(snapshot()).unwrap();
    graph.set_node_attr(4, "v".into(), V::Int(3)).unwrap();
    let mut graph = graph.into_publishing().unwrap();
    assert_eq!(graph.current_state_id(), 1);
    assert!(!graph.has_uncommitted_changes());
    let clock = graph.publication_metadata(1).unwrap().unwrap().effective_at;
    let old = graph.read_view_at(clock, None).unwrap();
    assert_eq!(old.node_ids(), vec![4, 7]);
    assert_eq!(old.edge_ids(), vec![9]);
    assert_eq!(
        old.node_attr(7, &"refs".into()).unwrap(),
        Some(V::NodeArray(vec![4, 7]))
    );
    graph
        .writer("test")
        .set_node_attr(4, "v".into(), V::Int(5))
        .unwrap();
    assert_eq!(
        graph
            .snapshot_at_commit(2)
            .unwrap()
            .node_attr(4, &"untouched".into()),
        Some(V::Bytes(vec![0, 255]))
    );
    assert_eq!(old.node_attr(4, &"v".into()).unwrap(), Some(V::Int(3)));
    let added = graph
        .apply_batch(
            GraphBatch {
                add_edges: vec![(N::Existing(4), N::Existing(7))],
                ..Default::default()
            },
            graph.current_state_id(),
            "edge".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(added.added_edges, vec![10]);
}
#[test]
fn malformed_snapshot_endpoints_reject_instead_of_creating_unrecorded_dangling_state() {
    let mut invalid = snapshot();
    invalid.edges.insert(10, (4, 99));
    assert!(Graph::from_snapshot(invalid).is_err());
}
