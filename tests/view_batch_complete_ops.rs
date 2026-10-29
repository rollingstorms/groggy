//! Selected write authorization and refreshed state cover the complete canonical batch.
use groggy::{AttrValue as V, BatchEdgeRef as E, BatchNodeRef as N, Graph, GraphBatch};
use std::collections::HashMap;
fn fixture() -> Graph {
    let mut graph = Graph::new_with_type(groggy::types::GraphType::Directed);
    let batch = GraphBatch {
        add_nodes: 4,
        add_edges: vec![
            (N::Added(0), N::Added(1)),
            (N::Added(0), N::Added(1)),
            (N::Added(2), N::Added(3)),
        ],
        node_attrs: HashMap::from([(
            "value".into(),
            (0..4).map(|id| (N::Added(id), V::Int(id as i64))).collect(),
        )]),
        edge_attrs: HashMap::from([(
            "weight".into(),
            (0..3)
                .map(|id| (E::Added(id), V::Int(id as i64 + 1)))
                .collect(),
        )]),
        effective_at: Some(0),
        ..Default::default()
    };
    graph
        .apply_batch(batch, 0, "fixture".into(), "test".into())
        .unwrap();
    graph
}
#[test]
fn unsets_and_endpoint_revisions_cannot_bypass_selected_targets_or_endpoints() {
    for operation in 0..4 {
        let mut graph = fixture();
        let source = graph.read_view().unwrap().select_edges(&[0]).unwrap();
        let head = graph.current_state_id();
        let mut batch = GraphBatch {
            add_nodes: 1,
            node_attrs: HashMap::from([("value".into(), vec![(N::Existing(0), V::Int(100))])]),
            effective_at: Some(10),
            ..Default::default()
        };
        match operation {
            0 => {
                batch
                    .unset_node_attrs
                    .insert("value".into(), vec![N::Existing(2)]);
            }
            1 => {
                batch
                    .unset_edge_attrs
                    .insert("weight".into(), vec![E::Existing(1)]);
            }
            2 => batch
                .edge_endpoints
                .push((1, N::Existing(0), N::Existing(1))),
            _ => batch
                .edge_endpoints
                .push((0, N::Existing(0), N::Existing(2))),
        }
        let error = graph
            .apply_view_batch(&source, batch, "outside".into(), "test".into())
            .unwrap_err();
        assert!(error.to_string().contains("outside"), "{error}");
        assert_eq!(graph.current_state_id(), head);
        assert_eq!(
            graph.get_node_attr(0, &"value".into()).unwrap(),
            Some(V::Int(0))
        );
        assert_eq!(
            graph.get_node_attr(2, &"value".into()).unwrap(),
            Some(V::Int(2))
        );
        assert_eq!(graph.edge_endpoints(0).unwrap(), (0, 1));
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(graph.add_node().unwrap(), 4);
    }
}
#[test]
fn complete_mixed_projection_preserves_rewired_edge_before_cascade_and_authorizes_followup() {
    let mut graph = fixture();
    let source = graph.read_view().unwrap().select_nodes(&[0, 1]).unwrap();
    let head = graph.current_state_id();
    let batch = GraphBatch {
        add_nodes: 1,
        add_edges: vec![(N::Added(0), N::Existing(0))],
        edge_endpoints: vec![(0, N::Existing(0), N::Added(0))],
        node_attrs: HashMap::from([("value".into(), vec![(N::Added(0), V::Null)])]),
        unset_node_attrs: HashMap::from([("value".into(), vec![N::Existing(0)])]),
        unset_edge_attrs: HashMap::from([("weight".into(), vec![E::Existing(0)])]),
        node_external_ids: vec![(N::Added(0), "new-node".into())],
        edge_external_ids: vec![(E::Existing(0), "kept-edge".into())],
        remove_nodes: vec![N::Existing(1)],
        effective_at: Some(10),
        ..Default::default()
    };
    let result = graph
        .apply_view_batch(&source, batch, "mixed".into(), "test".into())
        .unwrap();
    assert_eq!(result.batch.state_id, head + 1);
    assert_eq!(result.batch.added_nodes, vec![4]);
    assert_eq!(result.batch.added_edges, vec![3]);
    let fresh = result.view;
    assert_eq!(fresh.node_ids(), vec![0, 4]);
    assert_eq!(fresh.edge_ids(), vec![0, 3]);
    assert_eq!(fresh.edge_endpoints(0).unwrap(), (0, 4));
    assert_eq!(fresh.edge_endpoints(3).unwrap(), (4, 0));
    assert!(!fresh.has_node_attribute(0, &"value".into()).unwrap());
    assert!(!fresh.has_edge_attribute(0, &"weight".into()).unwrap());
    assert_eq!(fresh.node_attr(4, &"value".into()).unwrap(), Some(V::Null));
    assert_eq!(
        fresh.node_attr(4, &"external_id".into()).unwrap(),
        Some(V::Text("new-node".into()))
    );
    assert_eq!(
        fresh.edge_attr(0, &"external_id".into()).unwrap(),
        Some(V::Text("kept-edge".into()))
    );
    assert_eq!(source.edge_ids(), vec![0, 1]);
    assert_eq!(source.edge_endpoints(0).unwrap(), (0, 1));
    assert_eq!(
        source.node_attr(0, &"value".into()).unwrap(),
        Some(V::Int(0))
    );
    assert_eq!(
        graph
            .read_view_at(9, None)
            .unwrap()
            .edge_endpoints(0)
            .unwrap(),
        (0, 1)
    );
    assert_eq!(
        graph
            .read_view_at(10, None)
            .unwrap()
            .edge_endpoints(0)
            .unwrap(),
        (0, 4)
    );
    assert_eq!(
        graph
            .read_view_at(10, None)
            .unwrap()
            .node_attr(0, &"value".into())
            .unwrap(),
        None
    );
    let update = GraphBatch {
        edge_attrs: HashMap::from([("weight".into(), vec![(E::Existing(0), V::Int(8))])]),
        ..Default::default()
    };
    let next = graph
        .apply_view_batch(&fresh, update, "followup".into(), "test".into())
        .unwrap();
    assert_eq!(
        next.view.edge_attr(0, &"weight".into()).unwrap(),
        Some(V::Int(8))
    );
    assert_eq!(fresh.edge_attr(0, &"weight".into()).unwrap(), None);
    assert_eq!(graph.edge_endpoints(2).unwrap(), (2, 3));
}

#[test]
fn refreshed_views_keep_retained_meta_sources_after_owner_drop() {
    let mut graph = fixture();
    let group = graph
        .retain_group(&graph.read_view().unwrap(), "source".into())
        .unwrap();
    let meta = graph.create_meta_node(group).unwrap();
    let source = graph.read_view().unwrap().select_nodes(&[meta]).unwrap();
    let batch = GraphBatch {
        node_attrs: HashMap::from([(
            "caption".into(),
            vec![(N::Existing(meta), V::Text("changed".into()))],
        )]),
        ..Default::default()
    };
    let result = graph
        .apply_view_batch(&source, batch, "meta property".into(), "test".into())
        .unwrap();
    assert_eq!(
        result.view.meta_subgraph(meta).unwrap().node_ids(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        result.view.meta_subgraph(meta).unwrap().edge_ids(),
        vec![0, 1, 2]
    );
    drop(graph);
    assert_eq!(
        source.meta_subgraph(meta).unwrap().node_ids(),
        vec![0, 1, 2, 3]
    );
    std::thread::spawn(move || {
        assert_eq!(
            result
                .view
                .meta_subgraph(meta)
                .unwrap()
                .edge_endpoints(0)
                .unwrap(),
            (0, 1)
        );
    })
    .join()
    .unwrap();
}
