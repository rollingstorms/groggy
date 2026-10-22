use groggy::{
    storage::{
        adjacency::{MatrixFormat, MatrixType},
        GraphMatrix,
    },
    subgraphs::subgraph::Subgraph,
    types::AttrValue,
    Graph,
};
use std::{cell::RefCell, rc::Rc};
#[test]
fn edge_instances_sum_exactly_and_loops_contribute_once() {
    let edges = vec![(7, 9, 1e308), (7, 9, 1.), (7, 9, -1e308), (7, 7, 3.)];
    for directed in [false, true] {
        let sparse =
            GraphMatrix::sparse_weighted_adjacency_from_edges(&[9, 7], &edges, directed).unwrap();
        let dense =
            GraphMatrix::weighted_adjacency_from_edges_with_direction(&[9, 7], &edges, directed)
                .unwrap();
        assert!(sparse.is_sparse());
        assert!(!dense.is_sparse());
        assert_eq!(sparse.get(1, 0), Some(1.));
        assert_eq!(sparse.get(1, 1), Some(3.));
        assert_eq!(sparse.get(0, 1), Some(if directed { 0. } else { 1. }));
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(sparse.get(r, c), dense.get(r, c));
            }
        }
        assert_eq!(sparse.column_names(), ["node_9", "node_7"]);
    }
    let edges = [(0, 1, i64::MAX), (0, 1, 1), (0, 1, -i64::MAX)];
    let m = GraphMatrix::sparse_weighted_adjacency_from_edges(&[0, 1], &edges, true).unwrap();
    assert_eq!(m.get(0, 1), Some(1));
    assert!(GraphMatrix::sparse_weighted_adjacency_from_edges(&[0, 1], &edges[..2], true).is_err());
}
#[test]
fn invalid_coordinates_reject_and_rectangular_labels_match_rows() {
    assert!(GraphMatrix::<f64>::adjacency_from_edges(&[0, 0], &[]).is_err());
    assert!(GraphMatrix::<f64>::adjacency_from_edges(&[0], &[(0, 1)]).is_err());
    assert!(GraphMatrix::<f64>::adjacency_from_edges(&[], &[(0, 1)]).is_err());
    assert_eq!(
        GraphMatrix::<f64>::adjacency_from_edges(&[], &[])
            .unwrap()
            .shape(),
        (0, 0)
    );
    let m = GraphMatrix::from_row_major_data(vec![1.; 6], 2, 3, Some(&[7, 9])).unwrap();
    assert_eq!(m.column_names(), ["col_0", "col_1", "col_2"]);
    assert_eq!(m.row_labels().unwrap(), ["node_7", "node_9"]);
    assert!(GraphMatrix::from_row_major_data(vec![1.; 6], 2, 3, Some(&[7])).is_err());
}
#[test]
fn selected_edges_are_not_replaced_by_induced_parent_edges() {
    let mut g = Graph::new_directed();
    let a = g.add_node();
    let b = g.add_node();
    let e = g.add_edge(a, b).unwrap();
    let other = g.add_edge(a, b).unwrap();
    let loop_edge = g.add_edge(a, a).unwrap();
    g.set_edge_attr(e, "w".into(), AttrValue::Int(5)).unwrap();
    g.set_edge_attr(other, "w".into(), AttrValue::Int(99))
        .unwrap();
    let view = Subgraph::new(
        Rc::new(RefCell::new(g)),
        [a, b].into_iter().collect(),
        [e, loop_edge].into_iter().collect(),
        "test".into(),
    );
    for sparse in [false, true] {
        let m = view.adjacency_matrix(sparse, Some("w")).unwrap();
        assert_eq!(m.is_sparse(), sparse);
        assert_eq!(m.get(0, 1), Some(5.));
        assert_eq!(m.get(0, 0), Some(1.));
    }
}
#[test]
fn graph_sparse_exports_are_canonical_and_format_selection_is_real() {
    let mut g = Graph::new_directed();
    let nodes: Vec<_> = (0..10000).map(|_| g.add_node()).collect();
    for i in 0..nodes.len() {
        g.add_edge(nodes[i], nodes[(i + 1) % nodes.len()]).unwrap();
    }
    let m = g.sparse_adjacency_matrix().unwrap();
    assert!(m.is_sparse());
    assert_eq!(m.shape(), (10000, 10000));
    assert_eq!(m.get(0, 1), Some(1.));
    assert_eq!(m.get(9999, 0), Some(1.));
    assert_eq!(m.column_names()[9999], format!("node_{}", nodes[9999]));
    let p = g
        .custom_adjacency_matrix(MatrixFormat::Sparse, MatrixType::Transition, true, None)
        .unwrap();
    assert!(p.is_sparse());
    assert_eq!(p.get(0, 1), Some(1.));
    assert!(g
        .custom_adjacency_matrix(MatrixFormat::Sparse, MatrixType::Adjacency, false, None)
        .is_err());
}
#[test]
fn missing_weight_is_one_but_invalid_weight_does_not_disappear() {
    let mut g = Graph::new();
    let a = g.add_node();
    let b = g.add_node();
    let e = g.add_edge(a, b).unwrap();
    assert_eq!(
        g.weighted_adjacency_matrix("w").unwrap().get(0, 1),
        Some(1.)
    );
    g.set_edge_attr(e, "w".into(), AttrValue::Null).unwrap();
    assert_eq!(
        g.weighted_adjacency_matrix("w").unwrap().get(0, 1),
        Some(0.)
    );
    g.set_edge_attr(e, "w".into(), AttrValue::Text("invalid".into()))
        .unwrap();
    assert!(g.weighted_adjacency_matrix("w").is_err());
    assert!(g
        .adjacency_from_entities::<f64>(&[a, b], &[e, e], true, None)
        .is_err());
}
