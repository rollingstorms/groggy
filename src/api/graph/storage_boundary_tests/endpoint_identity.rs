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

#[test]
fn legacy_membership_rejects_revisions_before_mutation_and_sparse_storage_accepts_them() {
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 1).unwrap();
    graph
        .pool
        .borrow_mut()
        .store_subgraph(
            std::collections::HashSet::from([0, 1]),
            std::collections::HashSet::from([0]),
            "legacy".into(),
        )
        .unwrap();
    let bytes = graph.archive_document().unwrap().encode_document().unwrap();
    assert!(graph
        .set_edge_endpoints(0, 2, 1)
        .unwrap_err()
        .to_string()
        .contains("frozen"));
    assert_eq!(
        graph.archive_document().unwrap().encode_document().unwrap(),
        bytes
    );
    // Unattached registration has no retained source root and may be explicitly purged.
    graph.pool.borrow_mut().preserve_allocation(3, 100);
    graph.purge_unreferenced().unwrap();
    graph.set_edge_endpoints(0, 2, 1).unwrap();
    let copy = graph.prepare_archive().unwrap().restore().unwrap();
    assert_eq!(copy.edge_endpoints(0).unwrap(), (2, 1));
    copy.check_integrity().unwrap();
}

#[test]
fn retained_integrity_rejects_a_hashed_but_false_endpoint_before_value() {
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 1).unwrap();
    let parent = graph.current_state_id();
    let bad = graph
        .history
        .create_commit(
            crate::state::change_tracker::ChangeSet {
                edge_endpoint_changes: vec![crate::EdgeEndpointChange {
                    edge_id: 0,
                    old_endpoints: (1, 0),
                    new_endpoints: (2, 1),
                }],
                ..crate::state::change_tracker::ChangeSet::new()
            },
            "corrupt before".into(),
            "test".into(),
            Some(parent),
        )
        .unwrap();
    graph
        .history
        .update_branch_head(&"main".into(), bad)
        .unwrap();
    graph.current_commit = bad;
    graph
        .pool
        .borrow_mut()
        .restore_edge_revision(0, 2, 1)
        .unwrap();
    graph.space.activate_edge(0, 2, 1);
    assert!(graph
        .check_integrity()
        .unwrap_err()
        .to_string()
        .contains("before-state"));
}
