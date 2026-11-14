//! Native compatibility columns keep every row and share the attribute-first bulk reader.
use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, AttrValue as V, Graph};
use std::collections::HashSet;
use std::sync::Arc;
fn fixture() -> (Arc<OwnerCell<Graph>>, Subgraph) {
    let mut graph = Graph::new();
    graph.add_nodes(6).unwrap();
    graph.remove_node(1).unwrap();
    for (s, t) in [(0, 2), (0, 2), (2, 3)] {
        graph.add_edge(s, t).unwrap();
    }
    graph.set_node_attr(0, "v".into(), V::Float64(2.5)).unwrap();
    graph.set_node_attr(2, "v".into(), V::Null).unwrap();
    graph
        .set_node_attr(4, "v".into(), V::Int(9_007_199_254_740_993))
        .unwrap();
    graph
        .set_edge_attr(0, "v".into(), V::Float64Vec(vec![1., -0.]))
        .unwrap();
    graph.set_edge_attr(2, "v".into(), V::Null).unwrap();
    let graph = Arc::new(OwnerCell::new(graph));
    let selected = Subgraph::new(
        graph.clone(),
        HashSet::from([4, 0, 3, 2]),
        HashSet::from([2, 0]),
        "alignment".into(),
    );
    (graph, selected)
}
#[test]
fn columns_preserve_sorted_ids_typed_values_nulls_and_absent_rows() {
    let (graph, selected) = fixture();
    let head = graph.borrow().current_state_id();
    assert_eq!(selected.node_ids(), vec![0, 2, 3, 4]);
    assert_eq!(selected.edge_ids(), vec![0, 2]);
    assert_eq!(
        selected.get_node_attribute_column(&"v".into()).unwrap(),
        vec![
            V::Float64(2.5),
            V::Null,
            V::Null,
            V::Int(9_007_199_254_740_993)
        ]
    );
    assert_eq!(
        selected.get_edge_attribute_column(&"v".into()).unwrap(),
        vec![V::Float64Vec(vec![1., -0.]), V::Null]
    );
    assert_eq!(
        selected
            .get_node_attribute_column(&"absent".into())
            .unwrap(),
        vec![V::Null; 4]
    );
    assert_eq!(
        selected
            .get_edge_attribute_column(&"absent".into())
            .unwrap(),
        vec![V::Null; 2]
    );
    assert_eq!(graph.borrow().current_state_id(), head);
}
#[test]
fn explicit_read_sequences_keep_order_duplicates_and_missing_values_without_dropping() {
    let (_, selected) = fixture();
    assert_eq!(
        selected
            .get_node_attributes_for_nodes(&[4, 3, 2, 4], &"v".into())
            .unwrap(),
        vec![
            V::Int(9_007_199_254_740_993),
            V::Null,
            V::Null,
            V::Int(9_007_199_254_740_993)
        ]
    );
    assert!(selected
        .get_node_attributes_for_nodes(&[0, 5], &"v".into())
        .is_err());
    assert!(selected
        .get_node_attributes_for_nodes(&[0, 99], &"v".into())
        .is_err());
    assert!(selected
        .get_node_attributes_for_nodes(&[], &"v".into())
        .unwrap()
        .is_empty());
}
#[test]
fn inactive_rows_fail_instead_of_becoming_null_or_disappearing() {
    let (graph, selected) = fixture();
    graph.borrow_mut().remove_node(2).unwrap();
    let head = graph.borrow().current_state_id();
    assert!(selected.get_node_attribute_column(&"v".into()).is_err());
    assert!(selected.get_edge_attribute_column(&"v".into()).is_err());
    assert!(selected
        .get_node_attributes_for_nodes(&[0, 2], &"v".into())
        .is_err());
    assert_eq!(graph.borrow().current_state_id(), head);
}
#[test]
fn empty_columns_and_live_owner_updates_keep_the_existing_compatibility_boundary() {
    let (graph, selected) = fixture();
    let frozen = selected.pin_view().unwrap();
    graph
        .borrow_mut()
        .set_node_attr(3, "v".into(), V::Bool(true))
        .unwrap();
    assert_eq!(
        selected.get_node_attribute_column(&"v".into()).unwrap()[2],
        V::Bool(true)
    );
    assert_eq!(
        frozen.node_accessor().column(&"v".into()).get(2),
        Some(&V::Null)
    );
    let empty = Subgraph::new(graph, HashSet::new(), HashSet::new(), "empty".into());
    assert!(empty
        .get_node_attribute_column(&"v".into())
        .unwrap()
        .is_empty());
    assert!(empty
        .get_edge_attribute_column(&"v".into())
        .unwrap()
        .is_empty());
}
