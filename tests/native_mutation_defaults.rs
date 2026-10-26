use groggy::types::GraphType;
use groggy::{AttrValue as V, Graph, GraphConfig};
#[test]
fn all_ordinary_native_constructors_publish_fallible_node_and_bulk_operations() {
    let constructors = [
        Graph::new(),
        Graph::default(),
        Graph::new_directed(),
        Graph::new_undirected(),
        Graph::new_with_type(GraphType::Directed),
        Graph::with_config(GraphConfig::new()).unwrap(),
        Graph::new_publishing(),
    ];
    for mut graph in constructors {
        assert!(graph.automatic_publication());
        assert_eq!(graph.add_nodes(2).unwrap(), vec![0, 1]);
        let first = graph.current_state_id();
        assert_eq!(first, 1);
        assert!(!graph.has_uncommitted_changes());
        assert!(graph.publication_metadata(first).unwrap().is_some());
        let edge = graph.add_edge(0, 1).unwrap();
        assert_eq!(graph.current_state_id(), 2);
        let node = graph.add_node().unwrap();
        assert_eq!(node, 2);
        assert_eq!(graph.current_state_id(), 3);
        graph.set_node_attr(node, "v".into(), V::Int(7)).unwrap();
        assert_eq!(graph.current_state_id(), 4);
        assert!(!graph.has_uncommitted_changes());
        assert!(!graph.snapshot_at_commit(first).unwrap().edge_exists(edge));
        assert_eq!(
            graph
                .commit("explicit checkpoint".into(), "test".into())
                .unwrap(),
            4
        );
        assert_eq!(graph.history_records().len(), 4);
        assert_eq!(graph.checkpoints().len(), 1);
    }
}
#[test]
fn manual_constructors_are_explicit_and_keep_deferred_acceptance() {
    for mut graph in [
        Graph::new_manual(),
        Graph::new_directed_manual(),
        Graph::new_undirected_manual(),
        Graph::new_with_type_manual(GraphType::Directed),
        Graph::with_config_manual(GraphConfig::new()).unwrap(),
    ] {
        assert!(!graph.automatic_publication());
        assert_eq!(graph.add_node().unwrap(), 0);
        assert_eq!(graph.add_nodes(2).unwrap(), vec![1, 2]);
        assert_eq!(graph.current_state_id(), 0);
        assert!(graph.has_uncommitted_changes());
        assert!(graph.set_automatic_publication(true).is_err());
        assert_eq!(
            graph.commit("manual base".into(), "test".into()).unwrap(),
            1
        );
        assert_eq!(graph.publication_metadata(1).unwrap(), None);
        graph.set_automatic_publication(true).unwrap();
        assert_eq!(graph.add_node().unwrap(), 3);
        assert_eq!(graph.current_state_id(), 2);
        assert!(!graph.has_uncommitted_changes());
    }
}
#[test]
fn impossible_allocation_rejects_before_ids_or_history_in_both_policies() {
    for mut graph in [Graph::new(), Graph::new_manual()] {
        let state = graph.current_state_id();
        let before = graph.pool().statistics();
        assert!(graph.add_nodes(usize::MAX).is_err());
        assert_eq!(graph.current_state_id(), state);
        assert_eq!(graph.pool().statistics().node_count, before.node_count);
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.add_node().unwrap(), 0);
        let state = graph.current_state_id();
        let count = graph.node_count();
        assert!(graph.add_nodes(usize::MAX).is_err());
        assert_eq!(graph.node_count(), count);
        assert_eq!(graph.current_state_id(), state);
    }
}
#[test]
fn zero_bulk_has_no_allocation_or_publication_and_retains_existing_pending_changes() {
    for mut graph in [Graph::new(), Graph::new_manual()] {
        assert!(graph.add_nodes(0).unwrap().is_empty());
        assert_eq!(graph.current_state_id(), 0);
        assert!(!graph.has_uncommitted_changes());
        graph.add_node().unwrap();
        let state = graph.current_state_id();
        let dirty = graph.has_uncommitted_changes();
        assert!(graph.add_nodes(0).unwrap().is_empty());
        assert_eq!(graph.current_state_id(), state);
        assert_eq!(graph.has_uncommitted_changes(), dirty);
    }
}

#[test]
fn custom_configuration_rejects_invalid_inputs_before_construction_in_both_policies() {
    let mut invalid = GraphConfig::new();
    invalid.max_memory_usage = 0;
    assert!(Graph::with_config(invalid.clone()).is_err());
    assert!(Graph::with_config_manual(invalid).is_err());
    let mut invalid = GraphConfig::new();
    invalid.enable_query_cache = true;
    invalid.query_cache_size = 0;
    assert!(Graph::with_config(invalid.clone()).is_err());
    assert!(Graph::with_config_manual(invalid).is_err());
}
