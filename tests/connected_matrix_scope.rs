use groggy::{subgraphs::Subgraph, Graph};
use std::{cell::RefCell, rc::Rc};

fn fixture() -> Subgraph {
    let mut graph = Graph::new_directed();
    graph.add_nodes(4).unwrap();
    graph.add_edge(0, 1).unwrap();
    Subgraph::new(
        Rc::new(RefCell::new(graph)),
        (0..3).collect(),
        [0].into(),
        "selected".into(),
    )
}

#[test]
fn stale_node_or_edge_scope_rejects_for_dense_and_sparse_without_losing_frozen_rows() {
    for remove_node in [false, true] {
        let selected = fixture();
        let frozen = selected.pin_view().unwrap();
        let owned = selected.adjacency_matrix(true, None).unwrap();
        if remove_node {
            selected.graph().borrow_mut().remove_node(2).unwrap();
        } else {
            selected.graph().borrow_mut().remove_edge(0).unwrap();
        }
        for sparse in [false, true] {
            assert!(selected.adjacency_matrix(sparse, None).is_err());
            let old = frozen.adjacency_matrix::<f64>(sparse, None).unwrap();
            assert_eq!(old.shape(), (3, 3));
            assert_eq!(old.get(0, 1).unwrap(), 1.);
        }
        assert_eq!(owned.shape(), (3, 3));
        assert_eq!(owned.get(0, 1).unwrap(), 1.);
    }
}

#[test]
fn same_membership_endpoint_changes_invalidate_scope_and_inside_revisions_remain_visible() {
    let selected = fixture();
    assert_eq!(
        selected
            .adjacency_matrix(false, None)
            .unwrap()
            .get(0, 1)
            .unwrap(),
        1.
    );
    selected
        .graph()
        .borrow_mut()
        .set_edge_endpoints(0, 2, 1)
        .unwrap();
    let revised = selected.adjacency_matrix(true, None).unwrap();
    assert_eq!(revised.get(2, 1).unwrap(), 1.);
    assert_eq!(revised.get(0, 1).unwrap(), 0.);
    selected
        .graph()
        .borrow_mut()
        .set_edge_endpoints(0, 3, 1)
        .unwrap();
    for sparse in [false, true] {
        assert!(selected.adjacency_matrix(sparse, None).is_err());
    }
}
