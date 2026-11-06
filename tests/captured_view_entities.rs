//! Frozen entity/column access shares one capture and never borrows a working graph.
use groggy::{storage::array::BaseArray, AttrValue as V, Graph, GraphView};
use std::collections::HashSet;
fn fixture() -> (Graph, GraphView) {
    let mut graph = Graph::new_manual();
    graph.add_nodes(5).unwrap();
    graph.remove_node(1).unwrap();
    for (s, t) in [(0, 2), (0, 2), (2, 3), (0, 0)] {
        graph.add_edge(s, t).unwrap();
    }
    graph
        .set_node_attr(0, "score".into(), V::Float64(2.5))
        .unwrap();
    graph.set_node_attr(2, "score".into(), V::Null).unwrap();
    graph
        .set_node_attr(4, "score".into(), V::Int(9_007_199_254_740_993))
        .unwrap();
    graph
        .set_edge_attr(0, "weight".into(), V::Float64(2.))
        .unwrap();
    let root = graph.read_view().unwrap();
    let view = GraphView::from_membership(
        root.read_state().clone(),
        root.nodes().clone(),
        HashSet::from([0, 2]),
    )
    .unwrap();
    (graph, view)
}
#[test]
fn original_ids_and_sorted_positions_are_explicit_and_masks_keep_graph_closure() {
    let (_, view) = fixture();
    let nodes = view.node_accessor();
    assert_eq!(nodes.ids(), &[0, 2, 3, 4]);
    assert!(nodes.by_id(1).is_err());
    assert_eq!(nodes.by_id(2).unwrap().id(), 2);
    assert_eq!(nodes.at(1).unwrap().id(), 2);
    assert_eq!(nodes.at(-1).unwrap().id(), 4);
    assert!(nodes.at(isize::MIN).is_err());
    assert_eq!(
        nodes.slice_positions(3, -2, 2).unwrap().node_ids(),
        vec![2, 4]
    );
    assert!(nodes.slice_positions(4, 1, 1).is_err());
    assert!(nodes.slice_positions(0, 0, 1).is_err());
    let mask = BaseArray::new(vec![
        V::Bool(true),
        V::Bool(true),
        V::Bool(false),
        V::Bool(true),
    ]);
    let selected = nodes.select_mask(&mask).unwrap();
    assert_eq!(selected.node_ids(), vec![0, 2, 4]);
    assert_eq!(selected.edge_ids(), vec![0]);
    assert!(nodes
        .select_mask(&BaseArray::new(vec![V::Bool(true)]))
        .is_err());
    assert!(nodes
        .select_mask(&BaseArray::new(vec![V::Null; 4]))
        .is_err());
    assert_eq!(
        nodes
            .select_mask(&BaseArray::new(vec![V::Bool(false); 4]))
            .unwrap()
            .node_count(),
        0
    );
    assert!(nodes.select(&[99]).is_err());
    assert_eq!(nodes.select(&[2, 0, 2]).unwrap().edge_ids(), vec![0]);
}
#[test]
fn owned_columns_align_ids_and_preserve_explicit_null_vs_missing_and_exact_types() {
    let (_, view) = fixture();
    let nodes = view.node_accessor();
    let mut column = nodes.column(&"score".into());
    assert_eq!(
        column.iter().cloned().collect::<Vec<_>>(),
        vec![
            V::Float64(2.5),
            V::Null,
            V::Null,
            V::Int(9_007_199_254_740_993)
        ]
    );
    assert_eq!(
        nodes
            .presence(&"score".into())
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec![V::Bool(true), V::Bool(true), V::Bool(false), V::Bool(true)]
    );
    column.data_mut()[0] = V::Int(-1);
    assert_eq!(
        view.node_attr(0, &"score".into()).unwrap(),
        Some(V::Float64(2.5))
    );
    assert!(nodes.by_id(2).unwrap().contains_attribute(&"score".into()));
    assert!(!nodes.by_id(3).unwrap().contains_attribute(&"score".into()));
    assert_eq!(nodes.by_id(2).unwrap().to_dict()["score"], V::Null);
    assert_eq!(nodes.column(&"absent".into()).len(), 4);
    let edges = view.edge_accessor();
    assert_eq!(edges.ids(), &[0, 2]);
    assert_eq!(
        edges
            .column(&"weight".into())
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec![V::Float64(2.), V::Null]
    );
}
#[test]
fn handles_and_iterators_keep_exact_frozen_topology_and_outlive_owner() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<groggy::subgraphs::GraphViewEntity>();
    assert_send_sync::<groggy::subgraphs::GraphViewEntities>();
    let (mut graph, view) = fixture();
    let nodes = view.node_accessor();
    let node = nodes.by_id(0).unwrap();
    let edge = view.edge(0).unwrap();
    assert_eq!(node.degree().unwrap(), 1);
    assert_eq!(node.neighbors().unwrap(), vec![2]);
    assert_eq!(edge.endpoints().unwrap(), (0, 2));
    assert!(node.endpoints().is_err());
    assert!(edge.degree().is_err());
    assert!(view.node(1).is_err());
    assert!(view.edge(1).is_err());
    let mut iterator = nodes.iter();
    assert_eq!(iterator.len(), 4);
    assert_eq!(iterator.next().unwrap().id(), 0);
    assert_eq!(iterator.len(), 3);
    graph.set_edge_endpoints(0, 4, 3).unwrap();
    graph.remove_node(2).unwrap();
    drop(graph);
    drop(view);
    assert_eq!(
        iterator.map(|entity| entity.id()).collect::<Vec<_>>(),
        vec![2, 3, 4]
    );
    std::thread::spawn(move || {
        assert_eq!(edge.endpoints().unwrap(), (0, 2));
        assert_eq!(node.neighbors().unwrap(), vec![2]);
        assert_eq!(nodes.column(&"score".into()).len(), 4);
    })
    .join()
    .unwrap();
}
#[test]
fn node_all_keeps_isolates_while_edge_all_and_filters_keep_only_exact_instances() {
    let (_, view) = fixture();
    let nodes = view.node_accessor();
    let edges = view.edge_accessor();
    assert_eq!(nodes.all().unwrap().node_ids(), vec![0, 2, 3, 4]);
    assert_eq!(nodes.all().unwrap().edge_ids(), vec![0, 2]);
    assert_eq!(edges.all().unwrap().node_ids(), vec![0, 2, 3]);
    assert_eq!(edges.all().unwrap().edge_ids(), vec![0, 2]);
    let filter = groggy::query::NodeFilter::NodeSet(HashSet::from([0, 2]));
    assert_eq!(nodes.filter_nodes(&filter).unwrap().edge_ids(), vec![0]);
    assert!(edges.filter_nodes(&filter).is_err());
    assert_eq!(
        view.select_nodes(&[4])
            .unwrap()
            .edge_accessor()
            .all()
            .unwrap()
            .node_count(),
        0
    );
}
#[test]
fn captured_meta_entity_resolves_the_frozen_group_revision() {
    let mut graph = Graph::new();
    graph.add_nodes(2).unwrap();
    graph.add_edge(0, 1).unwrap();
    let source = graph.read_view().unwrap();
    let group = graph.retain_group(&source, "group".into()).unwrap();
    let meta = graph.create_meta_node(group).unwrap();
    let entity = graph.read_view().unwrap().node(meta).unwrap();
    let before = graph.current_state_id();
    assert_eq!(entity.meta_subgraph().unwrap().node_ids(), vec![0, 1]);
    assert_eq!(graph.current_state_id(), before);
    graph.remove_node(0).unwrap();
    drop(graph);
    assert_eq!(entity.meta_subgraph().unwrap().edge_ids(), vec![0]);
}

#[test]
fn parallel_column_and_presence_projection_preserve_exact_sorted_alignment() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(1001).unwrap();
    for id in 0..1001 {
        if id % 3 != 2 {
            graph
                .set_node_attr(
                    id,
                    "v".into(),
                    if id % 3 == 0 {
                        V::Int(id as i64)
                    } else {
                        V::Null
                    },
                )
                .unwrap();
        }
    }
    let view = graph.read_view().unwrap();
    let nodes = view.node_accessor();
    let values = nodes.column(&"v".into());
    let presence = nodes.presence(&"v".into());
    for (id, (value, present)) in values.iter().zip(presence.iter()).enumerate() {
        assert_eq!(
            value,
            &if id % 3 == 0 {
                V::Int(id as i64)
            } else {
                V::Null
            }
        );
        assert_eq!(present, &V::Bool(id % 3 != 2));
    }
    assert_eq!(nodes.select_mask(&presence).unwrap().node_count(), 668);
}

#[test]
fn present_columns_keep_explicit_null_and_original_order_after_owner_changes() {
    let (mut graph, view) = fixture();
    let nodes = view.node_accessor();
    assert_eq!(
        nodes.present_column(&"score".into()).data(),
        &vec![V::Float64(2.5), V::Null, V::Int(9_007_199_254_740_993)]
    );
    assert_eq!(
        nodes.column(&"score".into()).data(),
        &vec![
            V::Float64(2.5),
            V::Null,
            V::Null,
            V::Int(9_007_199_254_740_993)
        ]
    );
    graph.set_node_attr(0, "score".into(), V::Int(99)).unwrap();
    graph.remove_node(2).unwrap();
    assert_eq!(
        nodes.present_column(&"score".into()).data(),
        &vec![V::Float64(2.5), V::Null, V::Int(9_007_199_254_740_993)]
    );
    assert_eq!(
        view.edge_accessor().present_column(&"weight".into()).data(),
        &vec![V::Float64(2.)]
    );
    assert!(nodes.present_column(&"absent".into()).data().is_empty());
}
