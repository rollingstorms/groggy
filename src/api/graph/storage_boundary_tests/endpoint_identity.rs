use crate::types::GraphType;
use crate::Graph;

#[test]
fn raw_reconstruction_cannot_rewire_history_or_partially_replace_working_membership() {
    for sparse in [false, true] {
        for graph_type in [GraphType::Directed, GraphType::Undirected] {
            let config = crate::GraphConfig {
                graph_type,
                ..Default::default()
            };
            let mut graph = Graph::with_config(config).unwrap();
            graph.add_nodes(3).unwrap();
            let edge = graph.add_edge(0, 1).unwrap();
            if sparse {
                graph.pool.borrow_mut().preserve_allocation(3, 100);
                graph.purge_unreferenced().unwrap();
            }
            let before = graph.archive_document().unwrap().encode_document().unwrap();
            assert!(graph
                .pool
                .borrow_mut()
                .add_edge_with_id(edge, 1, 2)
                .is_err());
            assert_eq!(graph.pool.borrow().get_edge_endpoints(edge), Some((0, 1)));
            let mut proposed = graph
                .history
                .reconstruct_state_at(graph.current_state_id())
                .unwrap();
            proposed.active_nodes.push(999);
            proposed.edges.insert(edge, (1, 2));
            assert!(graph.reset_to_snapshot(proposed).is_err());
            assert_eq!(
                graph.archive_document().unwrap().encode_document().unwrap(),
                before
            );
            assert!(!graph.has_node(999));
            assert!(!graph.is_node_allocated(999));
            graph
                .pool
                .borrow_mut()
                .add_edge_with_id(edge, 0, 1)
                .unwrap();
            graph.check_integrity().unwrap();
        }
    }
}
