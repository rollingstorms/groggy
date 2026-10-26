//! Bulk cascades retain history, clocks and exact edge identity without per-node scans.
use groggy::{
    AttrValue, BatchChangeRef as T, BatchEdgeRef as E, BatchNodeRef as N, ChangeKey as K, Graph,
    GraphBatch,
};
use std::collections::{HashMap, HashSet};

#[test]
fn bulk_cascades_preserve_unselected_edges_and_immutable_history() {
    for directed in [false, true] {
        for publishing in [false, true] {
            let mut graph = if directed {
                Graph::new_directed_manual()
            } else {
                Graph::new_manual()
            };
            graph.set_automatic_publication(publishing).unwrap();
            let nodes = graph.insert_nodes(5).unwrap();
            let endpoints = [(0, 1), (1, 0), (1, 1), (0, 1), (1, 2), (3, 4), (4, 4)];
            let edges = graph
                .add_edges_with_attrs(
                    endpoints
                        .iter()
                        .enumerate()
                        .map(|(i, &(u, v))| {
                            (
                                u,
                                v,
                                HashMap::from([("weight".into(), AttrValue::Int(i as i64))]),
                            )
                        })
                        .collect(),
                )
                .unwrap();
            graph
                .set_node_attr(1, "name".into(), AttrValue::Text("retained".into()))
                .unwrap();
            let before = graph.commit("before".into(), "test".into()).unwrap();
            let pin = graph.read_view().unwrap();
            graph.remove_nodes(&nodes[..2]).unwrap();
            if publishing {
                assert_eq!(graph.current_state_id(), before + 1);
                assert!(!graph.has_uncommitted_changes());
                let meta = graph.publication_metadata(before + 1).unwrap().unwrap();
                assert_eq!(meta.change_times.len(), 7); // two nodes + five distinct edges
            } else {
                assert_eq!(graph.current_state_id(), before);
                assert!(graph.has_uncommitted_changes());
            }
            let after = graph.commit("after".into(), "test".into()).unwrap();
            assert_eq!(graph.node_ids_set(), HashSet::from([2, 3, 4]));
            assert_eq!(graph.edge_ids_set(), HashSet::from([edges[5], edges[6]]));
            let old = graph.snapshot_at_commit(before).unwrap();
            let new = graph.snapshot_at_commit(after).unwrap();
            for (i, &edge) in edges.iter().enumerate() {
                assert!(old.edge_exists(edge));
                assert_eq!(
                    old.edge_attr(edge, &"weight".into()),
                    Some(AttrValue::Int(i as i64))
                );
                assert_eq!(new.edge_exists(edge), i >= 5);
            }
            assert!(pin.contains_node(1));
            assert_eq!(
                old.node_attr(1, &"name".into()),
                Some(AttrValue::Text("retained".into()))
            );
            assert!(!new.node_exists(1));
            let version = graph.space().get_version();
            graph.remove_nodes(&[]).unwrap();
            graph.remove_edges(&[]).unwrap();
            assert_eq!(graph.space().get_version(), version);
            assert_eq!(graph.current_state_id(), after);
            assert!(!graph.has_uncommitted_changes());
        }
    }
}

#[test]
fn mixed_batch_cascade_clocks_preserve_explicit_edge_overrides() {
    let mut graph = Graph::new_publishing();
    graph.insert_nodes(4).unwrap();
    let edges = graph
        .add_edges(&[(0, 1), (0, 1), (1, 1), (1, 2), (2, 3)])
        .unwrap();
    let before = graph.current_state_id();
    let batch = GraphBatch {
        remove_nodes: vec![N::Existing(0), N::Existing(1)],
        remove_edges: vec![E::Existing(edges[1])],
        effective_at: Some(100),
        change_times: vec![
            (T::NodeRemoved(N::Existing(0)), 20),
            (T::NodeRemoved(N::Existing(1)), 10),
            (T::EdgeRemoved(E::Existing(edges[1])), 30),
        ],
        ..Default::default()
    };
    let result = graph
        .apply_batch(batch, before, "cascade".into(), "test".into())
        .unwrap();
    assert_eq!(result.state_id, before + 1);
    let metadata = graph
        .publication_metadata(result.state_id)
        .unwrap()
        .unwrap();
    assert_eq!(metadata.change_times.len(), 6);
    for &edge in &[edges[0], edges[2], edges[3]] {
        assert_eq!(metadata.change_times[&K::EdgeRemoved(edge)], 10);
    }
    assert_eq!(metadata.change_times[&K::EdgeRemoved(edges[1])], 30);
    assert!(!metadata
        .change_times
        .contains_key(&K::EdgeRemoved(edges[4])));
    assert_eq!(graph.edge_ids(), vec![edges[4]]);
    assert!(!graph.has_uncommitted_changes());
}

#[test]
fn isolated_node_changes_invalidate_cached_csr_and_preserve_old_arrays() {
    let mut graph = Graph::new_manual();
    let initial = graph.space().snapshot_csr(&graph.pool());
    assert!(initial.0.is_empty());
    let first = graph.add_node().unwrap();
    let one = graph.space().snapshot_csr(&graph.pool());
    assert_eq!(*one.0, vec![first]);
    let added = graph.add_nodes(3).unwrap();
    let four = graph.space().snapshot_csr(&graph.pool());
    assert_eq!(four.0.len(), 4);
    graph.remove_node(first).unwrap();
    let three = graph.space().snapshot_csr(&graph.pool());
    assert_eq!(*three.0, added);
    graph.remove_nodes(&added).unwrap();
    let empty = graph.space().snapshot_csr(&graph.pool());
    assert!(empty.0.is_empty());
    assert_eq!(*empty.1, vec![0]);
    assert_eq!(four.0.len(), 4);
    assert_eq!(*one.0, vec![first]);
}

#[test]
fn invalid_bulk_removal_keeps_cascade_properties_and_cache_unchanged() {
    for publishing in [false, true] {
        let mut graph = Graph::new_manual();
        graph.set_automatic_publication(publishing).unwrap();
        graph.insert_nodes(2).unwrap();
        graph.add_edge(0, 1).unwrap();
        graph.commit("before".into(), "test".into()).unwrap();
        let version = graph.space().get_version();
        let state = graph.current_state_id();
        for invalid in [vec![0, 0], vec![0, 99]] {
            assert!(graph.remove_nodes(&invalid).is_err());
            assert_eq!(graph.node_count(), 2);
            assert_eq!(graph.edge_count(), 1);
            assert_eq!(graph.space().get_version(), version);
            assert_eq!(graph.current_state_id(), state);
            assert!(!graph.has_uncommitted_changes());
        }
    }
}
