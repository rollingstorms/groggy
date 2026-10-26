//! A graph copy stages source validation, remapping and destination writes together.
use groggy::{AttrValue as V, CalendarDate, FixedDuration, Graph, UtcInstant};
use std::collections::{HashMap, HashSet};

fn destination(publishing: bool) -> Graph {
    let mut graph = Graph::new_manual();
    graph.set_automatic_publication(publishing).unwrap();
    graph.insert_nodes(3).unwrap();
    graph.add_edge(0, 0).unwrap();
    graph.remove_node(2).unwrap(); // allocation offset differs from active count
    graph.commit("existing".into(), "test".into()).unwrap();
    graph
}

#[test]
fn copy_remaps_typed_references_and_preserves_opaque_values() {
    for publishing in [false, true] {
        let mut source = Graph::new_directed_manual();
        source.add_nodes(4).unwrap();
        source.remove_node(0).unwrap();
        let source_edges = source.add_edges(&[(1, 2), (1, 2), (2, 2)]).unwrap();
        source
            .set_node_attr(3, "opaque".into(), V::IntVec(vec![0, 1, 2]))
            .unwrap();
        source
            .set_node_attr(1, "nodes".into(), V::NodeArray(vec![3, 1, 3]))
            .unwrap();
        source
            .set_node_attr(
                2,
                "edges".into(),
                V::EdgeArray(vec![source_edges[2], source_edges[0]]),
            )
            .unwrap();
        source
            .set_edge_attr(source_edges[0], "nodes".into(), V::NodeArray(vec![2, 1]))
            .unwrap();
        source
            .set_edge_attr(
                source_edges[2],
                "edges".into(),
                V::EdgeArray(vec![source_edges[1], source_edges[2]]),
            )
            .unwrap();
        let mut target = destination(publishing);
        let before = target.current_state_id();
        target.add_graph(&source).unwrap();
        assert_eq!(
            target.current_state_id(),
            before + usize::from(publishing) as u64
        );
        assert_eq!(target.automatic_publication(), publishing);
        assert_eq!(target.node_ids_set(), HashSet::from([0, 1, 3, 4, 5]));
        assert_eq!(
            target.get_node_attr(3, &"nodes".into()).unwrap(),
            Some(V::NodeArray(vec![5, 3, 5]))
        );
        assert_eq!(
            target.get_node_attr(4, &"edges".into()).unwrap(),
            Some(V::EdgeArray(vec![3, 1]))
        );
        assert_eq!(
            target.get_node_attr(5, &"opaque".into()).unwrap(),
            Some(V::IntVec(vec![0, 1, 2]))
        );
        assert_eq!(
            target.get_edge_attr(1, &"nodes".into()).unwrap(),
            Some(V::NodeArray(vec![4, 3]))
        );
        assert_eq!(
            target.get_edge_attr(3, &"edges".into()).unwrap(),
            Some(V::EdgeArray(vec![2, 3]))
        );
        assert_eq!(target.edge_endpoints(1).unwrap(), (3, 4));
        assert_eq!(target.edge_endpoints(2).unwrap(), (3, 4));
        assert_eq!(target.edge_endpoints(3).unwrap(), (4, 4));
        // The destination retains its graph direction; copying does not reconfigure it.
        assert!(target.is_undirected());
        let after = target.commit("copy".into(), "test".into()).unwrap();
        assert_eq!(
            target
                .snapshot_at_commit(after)
                .unwrap()
                .node_attr(3, &"nodes".into()),
            Some(V::NodeArray(vec![5, 3, 5]))
        );
        assert_eq!(
            source.get_node_attr(1, &"nodes".into()).unwrap(),
            Some(V::NodeArray(vec![3, 1, 3]))
        );
        assert!(source.has_uncommitted_changes()); // dirty source is captured, not committed
    }
}

#[test]
fn copy_preserves_supported_value_variants_and_null_vs_absence() {
    let values = [
        V::Null,
        V::Int(i64::MAX),
        V::Float64(f64::from_bits(0x3ff0000000000001)),
        V::Float64Vec(vec![-0.0, f64::INFINITY]),
        V::Bytes(vec![0, 255]),
        V::Json("{\"id\":1}".into()),
        V::TextVec(vec!["a".into(), "b".into()]),
        V::UtcInstant(UtcInstant::new(-1, 999_999).unwrap()),
        V::Date(CalendarDate::new(2000, 2, 29).unwrap()),
        V::Duration(FixedDuration::new(-1, 86_399, 999_999).unwrap()),
    ];
    for publishing in [false, true] {
        let mut source = Graph::new_manual();
        let nodes = source.add_nodes(values.len()).unwrap();
        let edges = source
            .add_edges(&nodes.iter().map(|&id| (id, id)).collect::<Vec<_>>())
            .unwrap();
        for (i, value) in values.iter().enumerate() {
            source
                .set_node_attr(nodes[i], "v".into(), value.clone())
                .unwrap();
            source
                .set_edge_attr(edges[i], "v".into(), value.clone())
                .unwrap();
        }
        let mut target = Graph::new_manual();
        target.set_automatic_publication(publishing).unwrap();
        target.add_graph(&source).unwrap();
        for (i, value) in values.iter().enumerate() {
            assert_eq!(
                target.get_node_attr(i, &"v".into()).unwrap(),
                Some(value.clone())
            );
            assert_eq!(
                target.get_edge_attr(i, &"v".into()).unwrap(),
                Some(value.clone())
            );
            assert_eq!(target.get_node_attr(i, &"missing".into()).unwrap(), None);
        }
    }
}

#[test]
fn unsupported_or_missing_references_reject_before_ids_or_pending_changes() {
    for publishing in [false, true] {
        for value in [
            V::NodeArray(vec![99]),
            V::EdgeArray(vec![99]),
            V::SubgraphRef(99),
        ] {
            for node_property in [false, true] {
                let mut source = Graph::new_manual();
                source.add_nodes(2).unwrap();
                source.add_edge(0, 1).unwrap();
                if node_property {
                    source
                        .set_node_attr(1, "refs".into(), value.clone())
                        .unwrap();
                } else {
                    source
                        .set_edge_attr(0, "refs".into(), value.clone())
                        .unwrap();
                }
                let mut target = destination(publishing);
                let before = target.current_state_id();
                assert!(target.add_graph(&source).is_err());
                assert_eq!(target.current_state_id(), before);
                assert!(!target.has_uncommitted_changes());
                assert_eq!(target.node_count(), 2);
                assert_eq!(target.edge_count(), 1);
                assert_eq!(target.insert_nodes(1).unwrap(), vec![3]);
                assert_eq!(target.add_edge(0, 1).unwrap(), 1);
            }
        }
    }
}

#[test]
fn corrupt_source_indices_and_kinds_cannot_be_silently_dropped() {
    for publishing in [false, true] {
        for corrupt_node in [false, true] {
            let mut source = Graph::new_manual();
            source.add_node().unwrap();
            source.add_edge(0, 0).unwrap();
            if corrupt_node {
                source
                    .space_mut()
                    .set_node_attr_index(0, "broken".into(), 99);
            } else {
                source
                    .space_mut()
                    .set_edge_attr_index(0, "broken".into(), 99);
            }
            let mut target = destination(publishing);
            assert!(target
                .add_graph(&source)
                .unwrap_err()
                .to_string()
                .contains("Missing retained value"));
            assert!(!target.has_uncommitted_changes());
            assert_eq!(target.insert_nodes(1).unwrap(), vec![3]);
        }
        let mut source = Graph::new_manual();
        source.add_node().unwrap();
        source
            .space_mut()
            .set_node_attr_index(0, "labels".into(), 0);
        // Point labels at the retained base marker: wrong logical type.
        source
            .pool_mut()
            .set_node_attribute(0, "labels".into(), V::Int(42))
            .unwrap();
        let mut target = destination(publishing);
        assert!(target.add_graph(&source).is_err());
        assert!(!target.has_uncommitted_changes());
        assert_eq!(target.insert_nodes(1).unwrap(), vec![3]);
    }
}

#[test]
fn identity_meta_and_dirty_target_boundaries_are_explicit() {
    for publishing in [false, true] {
        let mut source = Graph::new_publishing();
        source
            .upsert_nodes(vec![groggy::NodeUpsert {
                external_id: "reserved".into(),
                attrs: HashMap::new(),
            }])
            .unwrap();
        let mut target = destination(publishing);
        assert!(target
            .add_graph(&source)
            .unwrap_err()
            .to_string()
            .contains("explicit upserts"));
        assert_eq!(target.insert_nodes(1).unwrap(), vec![3]);

        let mut source = Graph::new_manual();
        source.add_node().unwrap();
        let reference = source
            .pool_mut()
            .store_subgraph(HashSet::from([0]), HashSet::new(), "subgraph".into())
            .unwrap();
        source.create_meta_node(reference).unwrap();
        let mut target = destination(publishing);
        assert!(target
            .add_graph(&source)
            .unwrap_err()
            .to_string()
            .contains("pooled/meta"));
        assert_eq!(target.insert_nodes(1).unwrap(), vec![3]);

        let mut source = Graph::new_manual();
        source.add_node().unwrap();
        let mut target = destination(publishing);
        target.set_automatic_publication(false).unwrap();
        target.add_node().unwrap(); // explicit pending manual edit
        if publishing {
            assert!(target.set_automatic_publication(true).is_err());
            assert!(target.try_add_node().is_err());
            assert_eq!(target.node_count(), 3);
        } else {
            target.add_graph(&source).unwrap();
            assert_eq!(target.node_count(), 4);
        }
        assert!(target.has_uncommitted_changes());
    }
}
