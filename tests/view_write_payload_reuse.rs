//! Accepted prepared outputs are reused weakly; source and catalog provenance stay separate.
use groggy::{AttrValue as V, BatchNodeRef as N, Graph, GraphBatch};
use std::collections::HashMap;
fn update() -> GraphBatch {
    GraphBatch {
        node_attrs: HashMap::from([("value".into(), vec![(N::Existing(0), V::Int(7))])]),
        ..Default::default()
    }
}
#[test]
fn accepted_output_shares_current_payload_but_keeps_original_and_invalidates_on_writes() {
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 1).unwrap();
    let original = graph.read_view().unwrap().select_nodes(&[0, 1]).unwrap();
    let result = graph
        .apply_view_batch(&original, update(), "update".into(), "test".into())
        .unwrap();
    let current = graph.read_view().unwrap();
    assert!(std::ptr::eq(
        current.read_state().snapshot(),
        result.view.read_state().snapshot()
    ));
    assert!(!std::ptr::eq(
        original.read_state().snapshot(),
        current.read_state().snapshot()
    ));
    assert_eq!(original.node_attr(0, &"value".into()).unwrap(), None);
    assert_eq!(
        current.node_attr(0, &"value".into()).unwrap(),
        Some(V::Int(7))
    );
    assert_eq!(current.node_ids(), vec![0, 1, 2]); // cached data is full; membership remains selected
    assert_eq!(result.view.node_ids(), vec![0, 1]);
    graph.set_node_attr(0, "value".into(), V::Int(9)).unwrap();
    let next = graph.read_view().unwrap();
    assert!(!std::ptr::eq(
        next.read_state().snapshot(),
        current.read_state().snapshot()
    ));
    assert_eq!(
        result.view.node_attr(0, &"value".into()).unwrap(),
        Some(V::Int(7))
    );
    assert_eq!(next.node_attr(0, &"value".into()).unwrap(), Some(V::Int(9)));
}
#[test]
fn sequential_scoped_writes_and_same_head_maintenance_use_correct_current_state() {
    let mut graph = Graph::new();
    graph.add_nodes(2).unwrap();
    graph.add_edge(0, 1).unwrap();
    let initial = graph.read_view().unwrap();
    let first = graph
        .apply_view_batch(&initial, update(), "first".into(), "test".into())
        .unwrap();
    // Physical maintenance may invalidate the cached payload without advancing the
    // logical source. An exact fallback comparison must still authorize this view.
    let head = graph.current_state_id();
    graph.purge_unreferenced().unwrap();
    assert_eq!(graph.current_state_id(), head);
    let maintained = graph.read_view().unwrap();
    assert!(!std::ptr::eq(
        maintained.read_state().snapshot(),
        first.view.read_state().snapshot()
    ));
    let batch = GraphBatch {
        unset_node_attrs: HashMap::from([("value".into(), vec![N::Existing(0)])]),
        ..Default::default()
    };
    let second = graph
        .apply_view_batch(&first.view, batch, "second".into(), "test".into())
        .unwrap();
    assert_eq!(second.view.node_attr(0, &"value".into()).unwrap(), None);
    assert_eq!(
        first.view.node_attr(0, &"value".into()).unwrap(),
        Some(V::Int(7))
    );
    assert!(graph
        .apply_view_batch(&initial, update(), "stale".into(), "test".into())
        .is_err());
    let current = graph.read_view().unwrap();
    assert!(std::ptr::eq(
        current.read_state().snapshot(),
        second.view.read_state().snapshot()
    ));
}
#[test]
fn new_group_catalog_is_rebound_even_when_frozen_values_reuse_current_payload() {
    let mut graph = Graph::new();
    graph.add_node().unwrap();
    let original = graph.read_view().unwrap();
    let mut child = Graph::new();
    child.add_nodes(2).unwrap();
    child.add_edge(0, 1).unwrap();
    let head = graph.current_state_id();
    let group = graph
        .retain_group(&child.read_view().unwrap(), "child".into())
        .unwrap();
    assert_eq!(graph.current_state_id(), head);
    assert!(original.read_state().retained_group(group).is_err());
    let batch = GraphBatch {
        node_attrs: HashMap::from([(
            "evidence".into(),
            vec![(N::Existing(0), V::SubgraphRef(group))],
        )]),
        ..Default::default()
    };
    let result = graph
        .apply_view_batch(&original, batch, "reference".into(), "test".into())
        .unwrap();
    let current = graph.read_view().unwrap();
    assert!(std::ptr::eq(
        current.read_state().snapshot(),
        result.view.read_state().snapshot()
    ));
    assert_eq!(
        current
            .read_state()
            .retained_group(group)
            .unwrap()
            .node_ids(),
        vec![0, 1]
    );
    assert_eq!(
        result
            .view
            .read_state()
            .retained_group(group)
            .unwrap()
            .edge_ids(),
        vec![0]
    );
    assert!(original.read_state().retained_group(group).is_err());
}

#[test]
fn durable_publication_reuses_prepared_payload_and_reopens_with_fresh_provenance() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("views.ggj");
    let mut graph = Graph::new();
    graph.add_nodes(2).unwrap();
    let result;
    {
        let mut store = groggy::DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let source = store.read_view().unwrap().select_nodes(&[0]).unwrap();
        result = store
            .apply_view_batch(&source, update(), "update".into(), "test".into())
            .unwrap();
        let current = store.read_view().unwrap();
        assert!(std::ptr::eq(
            current.read_state().snapshot(),
            result.view.read_state().snapshot()
        ));
        assert_eq!(current.node_ids(), vec![0, 1]);
        assert_eq!(result.view.node_ids(), vec![0]);
    }
    let mut reopened = groggy::DurableGraph::open(&path).unwrap();
    assert!(reopened
        .apply_view_batch(&result.view, update(), "foreign".into(), "test".into())
        .is_err());
    assert_eq!(
        reopened
            .read_view()
            .unwrap()
            .node_attr(0, &"value".into())
            .unwrap(),
        Some(V::Int(7))
    );
}
