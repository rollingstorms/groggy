//! Retained membership is independent of active state; meta creation is prepared.
use groggy::{AttrValue as V, ChangeKey as K, Graph};
use std::collections::HashSet;

#[test]
fn retained_group_creation_keeps_exact_required_properties_and_history() {
    for publishing in [false, true] {
        let mut graph = Graph::new_manual();
        graph.set_automatic_publication(publishing).unwrap();
        graph.insert_nodes(3).unwrap();
        let edge = graph.add_edge(0, 1).unwrap();
        let group = graph
            .pool_mut()
            .store_subgraph(
                HashSet::from([0, 1]),
                HashSet::from([edge]),
                "retained".into(),
            )
            .unwrap();
        graph.remove_nodes(&[0, 1]).unwrap();
        let before = graph
            .commit("inactive source".into(), "test".into())
            .unwrap();
        let old = graph.read_view().unwrap();
        let meta = graph.create_meta_node(group).unwrap();
        assert_eq!(meta, 3);
        assert_eq!(graph.current_state_id(), before + u64::from(publishing));
        assert_eq!(graph.has_uncommitted_changes(), !publishing);
        assert!(graph.is_meta_node(meta));
        assert!(!graph.is_base_node(meta));
        assert_eq!(graph.get_meta_nodes(), vec![meta]);
        assert_eq!(
            graph
                .get_node_attr(meta, &"contains_subgraph".into())
                .unwrap(),
            Some(V::SubgraphRef(group))
        );
        assert!(!old.contains_node(meta));
        assert_eq!(graph.pool().subgraph_count(), 1);
        let (nodes, edges, kind) = graph.pool().get_subgraph(group).unwrap();
        assert_eq!(nodes, HashSet::from([0, 1]));
        assert_eq!(edges, HashSet::from([edge]));
        assert_eq!(kind, "retained");
        if publishing {
            let metadata = graph.publication_metadata(before + 1).unwrap().unwrap();
            assert_eq!(metadata.change_times.len(), 3);
            for key in [
                K::NodeAdded(meta),
                K::NodeAttribute(meta, "entity_type".into()),
                K::NodeAttribute(meta, "contains_subgraph".into()),
            ] {
                assert_eq!(metadata.change_times[&key], metadata.recorded_at);
            }
        }
        let after = graph.commit("meta".into(), "test".into()).unwrap();
        let snapshot = graph.snapshot_at_commit(after).unwrap();
        assert_eq!(
            snapshot.node_attr(meta, &"entity_type".into()),
            Some(V::Text("meta".into()))
        );
        assert_eq!(
            snapshot.node_attr(meta, &"contains_subgraph".into()),
            Some(V::SubgraphRef(group))
        );
        assert!(!snapshot.node_exists(0));
    }
}

#[test]
fn invalid_group_and_dirty_publishing_head_preserve_allocations() {
    for publishing in [false, true] {
        let mut graph = Graph::new_manual();
        graph.set_automatic_publication(publishing).unwrap();
        graph.insert_nodes(1).unwrap();
        graph.commit("before".into(), "test".into()).unwrap();
        let before = graph.current_state_id();
        assert!(graph.create_meta_node(99).is_err());
        assert_eq!(graph.current_state_id(), before);
        assert_eq!(graph.pool().subgraph_count(), 0);
        assert!(!graph.has_uncommitted_changes());
        let group = graph
            .pool_mut()
            .store_subgraph(HashSet::from([0]), HashSet::new(), "group".into())
            .unwrap();
        graph.set_automatic_publication(false).unwrap();
        graph.add_node().unwrap(); // explicit deferred compatibility edit
        if publishing {
            assert!(graph.set_automatic_publication(true).is_err());
            assert!(graph.try_add_node().is_err());
            assert_eq!(graph.node_count(), 2);
            graph.commit("pending".into(), "test".into()).unwrap();
            graph.set_automatic_publication(true).unwrap();
        }
        assert_eq!(graph.create_meta_node(group).unwrap(), 2);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.pool().subgraph_count(), 1);
    }
}
