use groggy::Graph;
#[test]
fn zero_node_batches_preserve_ids_and_do_not_create_phantom_nodes() {
    let mut graph = Graph::new_manual();
    assert!(graph.add_nodes(0).unwrap().is_empty());
    assert!(graph.node_ids().is_empty());
    assert_eq!(graph.add_node().unwrap(), 0);
    assert!(graph.add_nodes(0).unwrap().is_empty());
    assert_eq!(graph.add_nodes(2).unwrap(), vec![1, 2]);
    assert_eq!(graph.node_ids().len(), 3);
}
#[test]
fn zero_pool_allocation_preserves_next_id_at_initial_and_nonzero_positions() {
    let mut graph = Graph::new_manual();
    let mut pool = graph.pool_mut();
    assert_eq!(pool.add_nodes_bulk(0), (0, 0, vec![]));
    assert_eq!(pool.add_node(), 0);
    assert_eq!(pool.add_nodes_bulk(0), (1, 1, vec![]));
    assert_eq!(pool.add_nodes_bulk(2), (1, 2, vec![1, 2]));
}
