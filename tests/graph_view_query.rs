use groggy::{
    query::{AttributeFilter as A, EdgeFilter as E, NodeFilter as N},
    subgraphs::Subgraph,
    types::GraphType,
    AttrValue as V, Graph,
};
use std::{cell::RefCell, collections::HashSet, rc::Rc};
fn set(ids: impl IntoIterator<Item = usize>) -> HashSet<usize> {
    ids.into_iter().collect()
}

#[test]
fn all_native_node_predicates_share_live_and_frozen_results() {
    let mut graph = Graph::new_with_type(GraphType::Directed);
    let n = graph.add_nodes(4);
    graph.add_edge(n[0], n[1]).unwrap();
    graph.add_edge(n[0], n[1]).unwrap();
    graph.add_edge(n[1], n[1]).unwrap();
    graph
        .set_node_attr(n[0], "score".into(), V::Int(i64::MAX))
        .unwrap();
    graph
        .set_node_attr(n[1], "score".into(), V::Int(i64::MAX - 1))
        .unwrap();
    graph.set_node_attr(n[2], "score".into(), V::Null).unwrap();
    let view = graph.read_view().unwrap();
    let cases = vec![
        (
            N::HasAttribute {
                name: "score".into(),
            },
            set([n[0], n[1], n[2]]),
        ),
        (
            N::AttributeEquals {
                name: "score".into(),
                value: V::Null,
            },
            set([n[2]]),
        ),
        (
            N::AttributeFilter {
                name: "score".into(),
                filter: A::GreaterThan(V::Int(i64::MAX - 1)),
            },
            set([n[0]]),
        ),
        (N::DegreeRange { min: 3, max: 3 }, set([n[1]])),
        (N::HasNeighbor { neighbor_id: n[0] }, set([n[1]])),
        (N::HasNeighbor { neighbor_id: n[1] }, set([n[0], n[1]])),
        (N::NodeSet(set([n[0], n[3], 9999])), set([n[0], n[3]])),
        (N::And(vec![]), set([])),
        (N::Or(vec![]), set([])),
        (
            N::Not(Box::new(N::HasAttribute {
                name: "score".into(),
            })),
            set([n[3]]),
        ),
        (
            N::And(vec![
                N::HasAttribute {
                    name: "score".into(),
                },
                N::Not(Box::new(N::AttributeEquals {
                    name: "score".into(),
                    value: V::Null,
                })),
            ]),
            set([n[0], n[1]]),
        ),
        (
            N::Or(vec![N::NodeSet(set([n[3]])), N::NodeSet(set([n[0], n[3]]))]),
            set([n[0], n[3]]),
        ),
    ];
    for (filter, expected) in cases {
        assert_eq!(
            set(graph.find_nodes(filter.clone()).unwrap()),
            expected,
            "live {filter:?}"
        );
        assert_eq!(
            set(view.filter_nodes(&filter).unwrap().node_ids()),
            expected,
            "frozen {filter:?}"
        );
    }
}

#[test]
fn scoped_queries_never_reintroduce_parent_nodes_or_excluded_parallel_instances() {
    for directed in [false, true] {
        let graph = Rc::new(RefCell::new(Graph::new_with_type(if directed {
            GraphType::Directed
        } else {
            GraphType::Undirected
        })));
        let mut g = graph.borrow_mut();
        let n = g.add_nodes(4);
        let edge = g.add_edge(n[0], n[1]).unwrap();
        let excluded = g.add_edge(n[0], n[1]).unwrap();
        let outside = g.add_edge(n[1], n[2]).unwrap();
        for id in &n {
            g.set_node_attr(*id, "keep".into(), V::Bool(true)).unwrap();
        }
        for id in [edge, excluded, outside] {
            g.set_edge_attr(id, "keep".into(), V::Bool(true)).unwrap();
        }
        let view = g.read_view().unwrap().select_edges(&[edge]).unwrap();
        drop(g);
        let legacy = Subgraph::new(
            graph.clone(),
            set([n[0], n[1]]),
            set([edge]),
            "scope".into(),
        );
        let yes = N::AttributeEquals {
            name: "keep".into(),
            value: V::Bool(true),
        };
        for ids in [
            view.filter_nodes(&yes).unwrap().edge_ids(),
            legacy.filter_nodes(&yes).unwrap().edge_ids(),
            legacy
                .filter_nodes_by_attribute(&"keep".into(), &V::Bool(true))
                .unwrap()
                .edge_ids(),
        ] {
            assert_eq!(set(ids), set([edge]));
        }
        assert_eq!(
            view.filter_nodes(&N::DegreeRange { min: 1, max: 1 })
                .unwrap()
                .node_count(),
            2
        );
        assert_eq!(
            legacy
                .filter_nodes(&N::DegreeRange { min: 1, max: 1 })
                .unwrap()
                .node_count(),
            2
        );
        assert!(view
            .filter_nodes(&N::HasNeighbor { neighbor_id: n[2] })
            .unwrap()
            .nodes()
            .is_empty());
        assert!(legacy
            .filter_nodes(&N::HasNeighbor { neighbor_id: n[2] })
            .unwrap()
            .nodes()
            .is_empty());
        let edge_yes = E::AttributeEquals {
            name: "keep".into(),
            value: V::Bool(true),
        };
        assert_eq!(view.filter_edges(&edge_yes).unwrap().edge_ids(), vec![edge]);
        assert_eq!(
            legacy.filter_edges(&edge_yes).unwrap().edges(),
            &set([edge])
        );
        let reverse = E::ConnectsNodes {
            source: n[1],
            target: n[0],
        };
        assert_eq!(
            view.filter_edges(&reverse).unwrap().edge_count(),
            usize::from(!directed)
        );
        assert_eq!(
            legacy.filter_edges(&reverse).unwrap().edge_count(),
            usize::from(!directed)
        );
        graph.borrow_mut().remove_node(n[0]).unwrap();
        assert!(legacy.filter_nodes(&yes).is_err());
        assert_eq!(view.filter_nodes(&yes).unwrap().node_count(), 2);
    }
}

#[test]
fn all_edge_predicates_keep_null_presence_boolean_rules_and_frozen_values() {
    let mut graph = Graph::new();
    let n = graph.add_nodes(3);
    let e = [
        graph.add_edge(n[0], n[1]).unwrap(),
        graph.add_edge(n[1], n[2]).unwrap(),
        graph.add_edge(n[2], n[2]).unwrap(),
    ];
    graph
        .set_edge_attr(e[0], "weight".into(), V::Int(7))
        .unwrap();
    graph.set_edge_attr(e[1], "weight".into(), V::Null).unwrap();
    let view = graph.read_view().unwrap();
    let cases = vec![
        (
            E::HasAttribute {
                name: "weight".into(),
            },
            set([e[0], e[1]]),
        ),
        (
            E::AttributeEquals {
                name: "weight".into(),
                value: V::Null,
            },
            set([e[1]]),
        ),
        (
            E::AttributeFilter {
                name: "weight".into(),
                filter: A::GreaterThan(V::SmallInt(6)),
            },
            set([e[0]]),
        ),
        (
            E::ConnectsNodes {
                source: n[1],
                target: n[0],
            },
            set([e[0]]),
        ),
        (E::ConnectsAny(vec![n[2], n[2]]), set([e[1], e[2]])),
        (E::And(vec![]), set(e)),
        (E::Or(vec![]), set([])),
        (
            E::Not(Box::new(E::HasAttribute {
                name: "weight".into(),
            })),
            set([e[2]]),
        ),
        (
            E::Or(vec![E::ConnectsAny(vec![n[0]]), E::ConnectsAny(vec![n[0]])]),
            set([e[0]]),
        ),
        (
            E::And(vec![
                E::HasAttribute {
                    name: "weight".into(),
                },
                E::Not(Box::new(E::AttributeEquals {
                    name: "weight".into(),
                    value: V::Null,
                })),
            ]),
            set([e[0]]),
        ),
    ];
    for (filter, expected) in cases {
        assert_eq!(set(graph.find_edges(filter.clone()).unwrap()), expected);
        assert_eq!(
            set(view.filter_edges(&filter).unwrap().edge_ids()),
            expected
        );
    }
    graph
        .set_edge_attr(e[0], "weight".into(), V::Int(-1))
        .unwrap();
    drop(graph);
    assert_eq!(
        view.filter_edges(&E::AttributeFilter {
            name: "weight".into(),
            filter: A::GreaterThan(V::Int(0))
        })
        .unwrap()
        .edge_ids(),
        vec![e[0]]
    );
}

#[test]
fn scalar_ranges_preserve_integer_order_mixed_boundaries_and_ieee_unordered_values() {
    assert!(A::GreaterThan(V::Int(i64::MAX - 1)).matches(&V::Int(i64::MAX)));
    assert!(A::LessThan(V::Float(2f32.powi(63))).matches(&V::Int(i64::MAX)));
    assert!(!A::GreaterThanOrEqual(V::Float(2f32.powi(63))).matches(&V::Int(i64::MAX)));
    assert!(A::GreaterThan(V::Int(i64::MIN)).matches(&V::Int(i64::MIN + 1)));
    assert!(A::LessThan(V::Float(f32::INFINITY)).matches(&V::Int(i64::MAX)));
    assert!(!A::LessThanOrEqual(V::Float(f32::NAN)).matches(&V::Int(0)));
    assert!(!A::GreaterThanOrEqual(V::Int(0)).matches(&V::Float(f32::NAN)));
    assert!(!A::GreaterThan(V::Int(0)).matches(&V::Bool(true)));
}

#[test]
fn parallel_attribute_columns_keep_sparse_rows_and_logical_candidate_context() {
    let mut graph = Graph::new();
    let nodes = graph.add_nodes(2001);
    for &node in nodes.iter().step_by(2) {
        graph
            .set_node_attr(node, "score".into(), V::Int(i64::MAX))
            .unwrap();
    }
    let predicate = N::AttributeFilter {
        name: "score".into(),
        filter: A::GreaterThan(V::Int(i64::MAX - 1)),
    };
    let expected: Vec<_> = nodes.iter().step_by(2).copied().collect();
    assert_eq!(
        set(graph.find_nodes(predicate.clone()).unwrap()),
        set(expected.clone())
    );
    assert_eq!(
        graph
            .read_view()
            .unwrap()
            .filter_nodes(&predicate)
            .unwrap()
            .node_ids(),
        expected
    );
}
