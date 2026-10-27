use groggy::{
    query::{AttributeFilter as A, EdgeFilter as E, NodeFilter as N},
    types::{CalendarDate, CompactString, FixedDuration, UtcInstant},
    AttrValue as V, BatchNodeRef as B, Graph, GraphBatch, GraphView,
};
use std::collections::HashMap;

fn values() -> Vec<V> {
    vec![
        V::Null,
        V::Bool(false),
        V::SmallInt(1),
        V::Int(1),
        V::Int(9_007_199_254_740_993),
        V::Int(i64::MAX),
        V::Int(i64::MIN),
        V::Float(1.),
        V::Float64(1.),
        V::Float64(9_007_199_254_740_992.),
        V::Float64(-0.),
        V::Float64(0.),
        V::Float64(f64::NAN),
        V::Float(f32::NAN),
        V::Float64(f64::NEG_INFINITY),
        V::Float64(f64::INFINITY),
        V::Text("alpha".into()),
        V::CompactText(CompactString::new("alpha")),
        V::Bytes(vec![1, 2]),
        V::Float64Vec(vec![1., 2.]),
        V::TextVec(vec!["x".into()]),
        V::Date(CalendarDate::new(2000, 2, 29).unwrap()),
        V::Date(CalendarDate::new(2024, 2, 29).unwrap()),
        V::UtcInstant(
            UtcInstant::from_calendar(CalendarDate::new(2024, 2, 29).unwrap(), 0, 0, 0, 0).unwrap(),
        ),
        V::Duration(FixedDuration::new(-1, 0, 0).unwrap()),
        V::Duration(FixedDuration::new(1, 0, 0).unwrap()),
    ]
}
fn fixture(directed: bool) -> Graph {
    let mut graph = if directed {
        Graph::new_directed()
    } else {
        Graph::new()
    };
    let values = values();
    let count = values.len() + 2;
    let batch = GraphBatch {
        add_nodes: count,
        add_edges: (0..count)
            .map(|id| (B::Added(id), B::Added((id + 1) % count)))
            .collect(),
        node_attrs: HashMap::from([(
            "x".into(),
            values
                .iter()
                .cloned()
                .enumerate()
                .map(|(id, v)| (B::Added(id), v))
                .collect(),
        )]),
        edge_attrs: HashMap::from([(
            "x".into(),
            values
                .iter()
                .cloned()
                .enumerate()
                .map(|(id, v)| (groggy::BatchEdgeRef::Added(id), v))
                .collect(),
        )]),
        effective_at: Some(10),
        ..Default::default()
    };
    graph
        .apply_batch(batch, 0, "fixture".into(), "test".into())
        .unwrap();
    graph
}
#[test]
fn indexed_predicates_match_scans_for_all_supported_domains_and_combinations() {
    let graph = fixture(false);
    for view in [
        graph.read_view().unwrap(),
        graph
            .read_view()
            .unwrap()
            .select_nodes(&(0..12).collect::<Vec<_>>())
            .unwrap(),
    ] {
        let index = view
            .build_index(
                &["x".into(), "absent".into()],
                &["x".into(), "absent".into()],
                true,
            )
            .unwrap();
        let mut nodes = vec![
            N::HasAttribute { name: "x".into() },
            N::HasAttribute {
                name: "absent".into(),
            },
            N::And(vec![]),
            N::Or(vec![]),
        ];
        let mut edges = vec![
            E::HasAttribute { name: "x".into() },
            E::HasAttribute {
                name: "absent".into(),
            },
            E::And(vec![]),
            E::Or(vec![]),
        ];
        for value in values() {
            nodes.push(N::AttributeEquals {
                name: "x".into(),
                value: value.clone(),
            });
            edges.push(E::AttributeEquals {
                name: "x".into(),
                value: value.clone(),
            });
            for filter in [
                A::Equals(value.clone()),
                A::NotEquals(value.clone()),
                A::GreaterThan(value.clone()),
                A::GreaterThanOrEqual(value.clone()),
                A::LessThan(value.clone()),
                A::LessThanOrEqual(value),
            ] {
                nodes.push(N::AttributeFilter {
                    name: "x".into(),
                    filter: filter.clone(),
                });
                edges.push(E::AttributeFilter {
                    name: "x".into(),
                    filter,
                });
            }
        }
        for filter in nodes.clone() {
            for filter in [
                filter.clone(),
                N::Not(Box::new(filter.clone())),
                N::And(vec![filter.clone(), N::HasAttribute { name: "x".into() }]),
                N::Or(vec![
                    filter,
                    N::HasAttribute {
                        name: "absent".into(),
                    },
                ]),
            ] {
                assert_eq!(
                    index.find_nodes(&filter, false).unwrap(),
                    view.filter_nodes(&filter).unwrap().node_ids(),
                    "{filter:?}"
                );
            }
        }
        for filter in edges.clone() {
            for filter in [
                filter.clone(),
                E::Not(Box::new(filter.clone())),
                E::And(vec![filter.clone(), E::HasAttribute { name: "x".into() }]),
                E::Or(vec![
                    filter,
                    E::HasAttribute {
                        name: "absent".into(),
                    },
                ]),
            ] {
                assert_eq!(
                    index.find_edges(&filter, false).unwrap(),
                    view.filter_edges(&filter).unwrap().edge_ids(),
                    "{filter:?}"
                );
            }
        }
    }
}
fn compare_topology(view: GraphView) {
    let index = view.build_index(&[], &[], true).unwrap();
    for n in 0..5 {
        let filter = N::HasNeighbor { neighbor_id: n };
        assert_eq!(
            index.find_nodes(&filter, false).unwrap(),
            view.filter_nodes(&filter).unwrap().node_ids()
        );
        for max in 0..8 {
            let filter = N::DegreeRange { min: n, max };
            assert_eq!(
                index.find_nodes(&filter, false).unwrap(),
                view.filter_nodes(&filter).unwrap().node_ids()
            );
        }
        for t in 0..5 {
            let filter = E::ConnectsNodes {
                source: n,
                target: t,
            };
            assert_eq!(
                index.find_edges(&filter, false).unwrap(),
                view.filter_edges(&filter).unwrap().edge_ids()
            );
            let filter = E::ConnectsAny(vec![n, t, n]);
            assert_eq!(
                index.find_edges(&filter, false).unwrap(),
                view.filter_edges(&filter).unwrap().edge_ids()
            );
        }
    }
}
#[test]
fn topology_postings_match_scans_with_direction_parallel_edges_loops_and_selection() {
    for directed in [false, true] {
        for mask in 0..512 {
            let mut graph = if directed {
                Graph::new_directed()
            } else {
                Graph::new()
            };
            let mut edges = Vec::new();
            for s in 0..3 {
                for t in 0..3 {
                    if mask & (1 << (s * 3 + t)) != 0 {
                        edges.push((B::Added(s), B::Added(t)));
                    }
                }
            }
            if mask & 1 != 0 {
                edges.push((B::Added(0), B::Added(0)));
            }
            graph
                .apply_batch(
                    GraphBatch {
                        add_nodes: 4,
                        add_edges: edges,
                        ..Default::default()
                    },
                    0,
                    "fixture".into(),
                    "test".into(),
                )
                .unwrap();
            let view = graph.read_view().unwrap();
            compare_topology(view.clone());
            compare_topology(view.select_nodes(&[0, 1, 3]).unwrap());
            compare_topology(
                view.select_edges(&view.edge_ids().into_iter().step_by(2).collect::<Vec<_>>())
                    .unwrap(),
            );
        }
    }
}
#[test]
fn missing_indexes_fail_before_boolean_short_circuit_and_scan_requires_explicit_choice() {
    let graph = fixture(false);
    let view = graph.read_view().unwrap();
    let index = view.build_index(&["x".into()], &[], false).unwrap();
    let filter = N::And(vec![
        N::NodeSet(Default::default()),
        N::HasAttribute {
            name: "missing".into(),
        },
    ]);
    assert!(index
        .find_nodes(&filter, false)
        .unwrap_err()
        .to_string()
        .contains("missing"));
    assert_eq!(
        index.find_nodes(&filter, true).unwrap(),
        view.filter_nodes(&filter).unwrap().node_ids()
    );
    assert!(index
        .find_nodes(&N::DegreeRange { min: 0, max: 0 }, false)
        .is_err());
    assert!(index.find_edges(&E::ConnectsAny(vec![]), false).is_err());
    assert!(view
        .build_index(&["x".into(), "x".into()], &[], false)
        .is_err());
    assert!(view.build_index(&[" ".into()], &[], false).is_err());
    let edge = E::HasAttribute { name: "x".into() };
    assert!(index.find_edges(&edge, false).is_err());
    assert_eq!(
        index.find_edges(&edge, true).unwrap(),
        view.filter_edges(&edge).unwrap().edge_ids()
    );
}
#[test]
fn captures_are_immutable_across_mutation_branch_switch_rejection_and_owner_drop() {
    let mut graph = fixture(false);
    graph
        .set_node_labels(0, vec!["Person".into(), "Evidence".into()])
        .unwrap();
    graph
        .set_edge_relationship_type(0, Some("supports".into()))
        .unwrap();
    let index = graph
        .read_view()
        .unwrap()
        .build_index(
            &["x".into(), "labels".into()],
            &["relationship_type".into()],
            true,
        )
        .unwrap();
    let clone = index.clone();
    let filter = E::AttributeEquals {
        name: "relationship_type".into(),
        value: V::Text("supports".into()),
    };
    let captured = index.filter_edges(&filter, false).unwrap();
    graph.create_branch("before".into()).unwrap();
    graph.remove_node(0).unwrap();
    assert_eq!(
        index.nodes_with_label("Person").unwrap().node_ids(),
        vec![0]
    );
    assert_eq!(index.find_edges(&filter, false).unwrap(), vec![0]);
    let refreshed = index.rebuild(graph.read_view().unwrap()).unwrap();
    assert!(refreshed
        .nodes_with_label("Person")
        .unwrap()
        .node_ids()
        .is_empty());
    assert!(refreshed.find_edges(&filter, false).unwrap().is_empty());
    graph.checkout_branch("before".into()).unwrap();
    assert_eq!(
        index
            .rebuild(graph.read_view().unwrap())
            .unwrap()
            .find_edges(&filter, false)
            .unwrap(),
        vec![0]
    );
    drop(graph);
    drop(index);
    assert_eq!(clone.find_edges(&filter, false).unwrap(), vec![0]);
    assert_eq!(captured.edge_endpoints(0).unwrap(), (0, 1));
    assert_eq!(
        clone.node_properties(),
        vec!["labels".to_string(), "x".to_string()]
    );
}
#[test]
fn effective_time_index_keeps_its_exact_clocks_and_empty_captures_are_valid() {
    let mut graph = fixture(false);
    graph
        .writer("test")
        .with_effective_at(Some(20))
        .set_node_attr(0, "x".into(), V::Int(5))
        .unwrap();
    let old = graph.read_view_at(19, None).unwrap();
    let index = old.build_index(&["x".into()], &[], false).unwrap();
    let filter = N::AttributeEquals {
        name: "x".into(),
        value: V::Null,
    };
    let result = index.filter_nodes(&filter, false).unwrap();
    assert_eq!(result.node_ids(), vec![0]);
    assert_eq!(
        result.read_state().read_time(),
        old.read_state().read_time()
    );
    assert!(index
        .rebuild(graph.read_view_at(20, None).unwrap())
        .unwrap()
        .find_nodes(&filter, false)
        .unwrap()
        .is_empty());
    let empty = Graph::new()
        .read_view()
        .unwrap()
        .build_index(&["x".into()], &["x".into()], true)
        .unwrap();
    assert!(empty.find_nodes(&filter, false).unwrap().is_empty());
    assert!(empty
        .find_nodes(&N::DegreeRange { min: 0, max: 0 }, false)
        .unwrap()
        .is_empty());
}
