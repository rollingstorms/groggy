use groggy::{
    query::{AttributeFilter, EdgeFilter, NodeFilter, QueryOptions, TraversalDirection as D},
    types::{CalendarDate, CompactString},
    AttrValue as V, BatchEdgeRef as E, BatchNodeRef as N, Graph, GraphBatch,
};
use std::collections::HashMap;

fn all() -> NodeFilter {
    NodeFilter::Not(Box::new(NodeFilter::NodeSet(Default::default())))
}
fn query(values: Vec<V>) -> groggy::query::GraphQueryIndex {
    let mut graph = Graph::new();
    let count = values.len() + 1;
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: count,
                add_edges: (0..count).map(|i| (N::Added(i), N::Added(i))).collect(),
                node_attrs: HashMap::from([(
                    "x".into(),
                    values
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(i, v)| (N::Added(i), v))
                        .collect(),
                )]),
                edge_attrs: HashMap::from([(
                    "x".into(),
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| (E::Added(i), v))
                        .collect(),
                )]),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "test".into(),
        )
        .unwrap();
    graph
        .read_view()
        .unwrap()
        .build_index(&["x".into()], &["x".into()], true)
        .unwrap()
}
#[test]
fn exact_order_missing_last_and_id_ties_are_shared_by_nodes_edges_and_pages() {
    let index = query(vec![
        V::Null,
        V::Int(9_007_199_254_740_993),
        V::Float64(9_007_199_254_740_992.),
        V::Int(1),
        V::Float(1.),
        V::Float64(f64::NEG_INFINITY),
        V::Float64(f64::INFINITY),
    ]);
    for (descending, expected) in [
        (false, vec![5, 3, 4, 2, 1, 6, 0, 7]),
        (true, vec![6, 1, 2, 3, 4, 5, 0, 7]),
    ] {
        for offset in [0, 1, 4, 8, usize::MAX] {
            for limit in [None, Some(0), Some(1), Some(3), Some(usize::MAX)] {
                let options = QueryOptions {
                    order_by: Some("x".into()),
                    descending,
                    offset,
                    limit,
                };
                let page = expected
                    .iter()
                    .copied()
                    .skip(offset)
                    .take(limit.unwrap_or(usize::MAX))
                    .collect::<Vec<_>>();
                assert_eq!(index.query_nodes(&all(), &options, false).unwrap(), page);
                assert_eq!(
                    index
                        .query_edges(&EdgeFilter::And(vec![]), &options, false)
                        .unwrap(),
                    page
                );
            }
        }
    }
    let plain = QueryOptions {
        descending: true,
        offset: 2,
        limit: Some(3),
        ..Default::default()
    };
    assert_eq!(
        index.query_nodes(&all(), &plain, false).unwrap(),
        vec![5, 4, 3]
    );
    assert_eq!(
        index
            .query_nodes(&all(), &QueryOptions::default(), false)
            .unwrap(),
        index.find_nodes(&all(), false).unwrap()
    );
}
#[test]
fn scalar_domains_are_checked_before_empty_pages_and_index_fallback_is_explicit() {
    let options = QueryOptions {
        order_by: Some("x".into()),
        limit: Some(0),
        ..Default::default()
    };
    for values in [
        vec![V::Float64(f64::NAN)],
        vec![V::Int(1), V::Text("x".into())],
        vec![V::Float64Vec(vec![1.])],
        vec![V::Bool(true), V::Int(1)],
    ] {
        let index = query(values);
        assert!(index.query_nodes(&all(), &options, false).is_err());
        assert!(index
            .query_edges(&EdgeFilter::And(vec![]), &options, false)
            .is_err());
    }
    let index = query(vec![V::Int(1)]);
    let missing = QueryOptions {
        order_by: Some("absent".into()),
        ..Default::default()
    };
    assert!(index
        .query_nodes(&NodeFilter::NodeSet(Default::default()), &missing, false)
        .is_err());
    assert_eq!(
        index.query_nodes(&all(), &missing, true).unwrap(),
        vec![0, 1]
    );
    let absent = index
        .view()
        .build_index(&["absent".into()], &[], false)
        .unwrap();
    assert_eq!(
        absent.query_nodes(&all(), &missing, false).unwrap(),
        vec![0, 1]
    );
    assert!(index
        .query_nodes(
            &all(),
            &QueryOptions {
                order_by: Some(" ".into()),
                ..Default::default()
            },
            true
        )
        .is_err());
    let filtered = NodeFilter::AttributeFilter {
        name: "x".into(),
        filter: AttributeFilter::GreaterThan(V::Int(0)),
    };
    assert_eq!(
        index
            .query_nodes(
                &filtered,
                &QueryOptions {
                    order_by: Some("x".into()),
                    ..Default::default()
                },
                false
            )
            .unwrap(),
        vec![0]
    );
}
#[test]
fn text_bool_and_temporal_sorting_follow_scalar_domains_and_nulls() {
    for (values, expected) in [
        (
            vec![
                V::Text("z".into()),
                V::CompactText(CompactString::new("a")),
                V::CompressedText(groggy::types::CompressedData::compress_text("a")),
                V::Null,
            ],
            vec![1, 2, 0, 3, 4],
        ),
        (
            vec![V::Bool(true), V::Bool(false), V::Null],
            vec![1, 0, 2, 3],
        ),
        (
            vec![
                V::Date(CalendarDate::new(2024, 2, 29).unwrap()),
                V::Date(CalendarDate::new(2000, 1, 1).unwrap()),
            ],
            vec![1, 0, 2],
        ),
    ] {
        assert_eq!(
            query(values)
                .query_nodes(
                    &all(),
                    &QueryOptions {
                        order_by: Some("x".into()),
                        ..Default::default()
                    },
                    false
                )
                .unwrap(),
            expected
        );
    }
}
#[test]
fn bounded_directional_selections_match_all_pairs_reference_on_all_three_node_graphs() {
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
                        edges.push((N::Added(s), N::Added(t)));
                    }
                }
            }
            if mask & 1 != 0 {
                edges.push((N::Added(0), N::Added(0)));
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
            for selected in [
                view.clone(),
                view.select_nodes(&[0, 1, 3]).unwrap(),
                view.select_edges(&view.edge_ids().into_iter().step_by(2).collect::<Vec<_>>())
                    .unwrap(),
            ] {
                for direction in [D::Outgoing, D::Incoming, D::Both] {
                    let mut distances = [[100usize; 4]; 4];
                    for (n, row) in distances.iter_mut().enumerate() {
                        row[n] = 0;
                    }
                    for edge in selected.edge_ids() {
                        let (s, t) = selected.edge_endpoints(edge).unwrap();
                        if !directed || direction != D::Incoming {
                            distances[s][t] = distances[s][t].min(1);
                        }
                        if !directed || direction != D::Outgoing {
                            distances[t][s] = distances[t][s].min(1);
                        }
                    }
                    for k in 0..4 {
                        for i in 0..4 {
                            for j in 0..4 {
                                distances[i][j] =
                                    distances[i][j].min(distances[i][k] + distances[k][j]);
                            }
                        }
                    }
                    for &start in &selected.node_ids() {
                        for hops in 0..5 {
                            let expected: Vec<_> = selected
                                .node_ids()
                                .into_iter()
                                .filter(|&n| distances[start][n] <= hops)
                                .collect();
                            let result = selected
                                .within_hops(&[start, start], hops, direction)
                                .unwrap();
                            assert_eq!(result.node_ids(), expected);
                            assert_eq!(
                                result.edge_ids(),
                                selected.select_nodes(&expected).unwrap().edge_ids()
                            );
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn bounded_reads_validate_all_seeds_and_preserve_time_source_and_owner_independence() {
    let mut graph = Graph::new_directed();
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 4,
                add_edges: vec![(N::Added(0), N::Added(1)), (N::Added(1), N::Added(2))],
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "test".into(),
        )
        .unwrap();
    let pin = graph.read_view_at(15, None).unwrap();
    let state = graph.current_state_id();
    assert!(pin.within_hops(&[0, 99], 0, D::Outgoing).is_err());
    assert!(pin
        .within_hops(&[], usize::MAX, D::Both)
        .unwrap()
        .node_ids()
        .is_empty());
    assert_eq!(
        pin.within_hops(&[0, 3], 1, D::Outgoing).unwrap().node_ids(),
        vec![0, 1, 3]
    );
    let selected = pin.within_hops(&[2], usize::MAX, D::Incoming).unwrap();
    assert_eq!(selected.node_ids(), vec![0, 1, 2]);
    assert_eq!(
        selected.read_state().read_time(),
        pin.read_state().read_time()
    );
    assert!(selected.write_state_id().is_none());
    assert_eq!(graph.current_state_id(), state);
    graph.remove_node(1).unwrap();
    drop(graph);
    assert_eq!(
        selected
            .within_hops(&[0], 2, D::Outgoing)
            .unwrap()
            .node_ids(),
        vec![0, 1, 2]
    );
}

#[test]
fn direct_posting_pages_match_scans_for_typed_values_scopes_and_extreme_bounds() {
    let values = vec![
        V::Int(1),
        V::SmallInt(1),
        V::Float(1.0),
        V::Float64(1.0),
        V::Null,
        V::Text("one".into()),
        V::CompactText(CompactString::new("one")),
        V::Int(1),
        V::Null,
        V::Float64(f64::NAN),
        V::Bytes(vec![1, 2]),
    ];
    let full = query(values.clone());
    let selected = full
        .view()
        .select_edges(&[0, 2, 4, 7])
        .unwrap()
        .build_index(&["x".into()], &["x".into()], false)
        .unwrap();
    for index in [&full, &selected] {
        let mut node_filters = vec![NodeFilter::HasAttribute { name: "x".into() }];
        let mut edge_filters = vec![EdgeFilter::HasAttribute { name: "x".into() }];
        for value in values.iter().chain([V::Int(-9)].iter()) {
            node_filters.push(NodeFilter::AttributeEquals {
                name: "x".into(),
                value: value.clone(),
            });
            node_filters.push(NodeFilter::AttributeFilter {
                name: "x".into(),
                filter: AttributeFilter::Equals(value.clone()),
            });
            edge_filters.push(EdgeFilter::AttributeEquals {
                name: "x".into(),
                value: value.clone(),
            });
            edge_filters.push(EdgeFilter::AttributeFilter {
                name: "x".into(),
                filter: AttributeFilter::Equals(value.clone()),
            });
        }
        for descending in [false, true] {
            for offset in [0, 1, 4, usize::MAX] {
                for limit in [None, Some(0), Some(1), Some(3), Some(usize::MAX)] {
                    let options = QueryOptions {
                        descending,
                        offset,
                        limit,
                        ..Default::default()
                    };
                    for filter in &node_filters {
                        assert_eq!(
                            index.query_nodes(filter, &options, false).unwrap(),
                            index.view().query_nodes(filter, &options).unwrap()
                        );
                    }
                    for filter in &edge_filters {
                        assert_eq!(
                            index.query_edges(filter, &options, false).unwrap(),
                            index.view().query_edges(filter, &options).unwrap()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn direct_pages_preserve_missing_index_and_full_scalar_order_validation() {
    let index = query(vec![V::Bytes(vec![1, 2]), V::Null]);
    let node = NodeFilter::HasAttribute { name: "x".into() };
    let edge = EdgeFilter::HasAttribute { name: "x".into() };
    let absent = index.view().build_index(&[], &[], false).unwrap();
    let options = QueryOptions {
        offset: usize::MAX,
        limit: Some(0),
        ..Default::default()
    };
    assert!(absent.query_nodes(&node, &options, false).is_err());
    assert!(absent.query_edges(&edge, &options, false).is_err());
    assert!(absent
        .query_nodes(&node, &options, true)
        .unwrap()
        .is_empty());
    assert!(absent
        .query_edges(&edge, &options, true)
        .unwrap()
        .is_empty());
    let ordered = QueryOptions {
        order_by: Some("x".into()),
        ..options
    };
    assert!(index.query_nodes(&node, &ordered, false).is_err());
    assert!(index.query_edges(&edge, &ordered, false).is_err());
}
