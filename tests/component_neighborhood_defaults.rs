use groggy::{query::selection::TraversalDirection as D, Graph};

#[test]
fn first_center_is_lowest_id_and_keeps_exact_captured_internal_instances() {
    let mut graph = Graph::new_directed();
    graph.add_nodes(6).unwrap();
    graph
        .add_edges(&[(0, 1), (1, 2), (2, 3), (4, 5), (1, 1), (0, 1)])
        .unwrap();
    let source = graph
        .read_view()
        .unwrap()
        .select_nodes(&[3, 1, 0, 2])
        .unwrap();
    graph.remove_node(0).unwrap();
    let neighbors = source.neighborhood_from_first(1, D::Both).unwrap();
    assert_eq!(neighbors.node_ids(), vec![0, 1]);
    assert_eq!(neighbors.edge_ids(), vec![0, 4, 5]);
    assert!(neighbors.is_directed());
    assert!(source.merge_with(&neighbors).is_ok());
    assert_eq!(
        source
            .neighborhood_from_first(0, D::Both)
            .unwrap()
            .node_ids(),
        vec![0]
    );
}

#[test]
fn first_center_obeys_explicit_direction_and_empty_source_identity() {
    let mut graph = Graph::new_directed();
    graph.add_nodes(3).unwrap();
    graph.add_edges(&[(1, 0), (0, 2)]).unwrap();
    let source = graph.read_view().unwrap();
    assert_eq!(
        source
            .neighborhood_from_first(1, D::Incoming)
            .unwrap()
            .node_ids(),
        vec![0, 1]
    );
    assert_eq!(
        source
            .neighborhood_from_first(1, D::Outgoing)
            .unwrap()
            .node_ids(),
        vec![0, 2]
    );
    let empty = source.select_nodes(&[]).unwrap();
    let result = empty.neighborhood_from_first(1, D::Both).unwrap();
    assert!(result.is_directed());
    assert_eq!(result.node_count(), 0);
    assert!(source.merge_with(&result).is_ok());
}
