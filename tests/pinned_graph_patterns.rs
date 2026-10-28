use groggy::{
    query::{
        EdgeFilter, GraphPattern, NodeFilter, PatternEdge, PatternMatch, PatternNode,
        PatternOptions, TraversalDirection as D,
    },
    AttrValue as V, BatchNodeRef as N, Graph, GraphBatch, GraphView,
};
use std::collections::{HashMap, HashSet};
fn node(name: &str) -> PatternNode {
    PatternNode {
        variable: name.into(),
        filter: None,
    }
}
fn edge(name: &str, a: &str, b: &str, direction: D) -> PatternEdge {
    PatternEdge {
        variable: name.into(),
        source: a.into(),
        target: b.into(),
        direction,
        filter: None,
    }
}
fn path(direction: D) -> GraphPattern {
    GraphPattern {
        nodes: vec![node("a"), node("b"), node("c")],
        edges: vec![
            edge("ab", "a", "b", direction),
            edge("bc", "b", "c", direction),
        ],
    }
}
fn decode(mut ordinal: usize, domain: &[usize], width: usize) -> Vec<usize> {
    (0..width)
        .map(|_| {
            let id = domain[ordinal % domain.len()];
            ordinal /= domain.len();
            id
        })
        .collect()
}
// Independent complete Cartesian enumeration: no adjacency joins or pruning.
fn brute(view: &GraphView, pattern: &GraphPattern, options: &PatternOptions) -> Vec<PatternMatch> {
    let nodes = view.node_ids();
    let edges = view.edge_ids();
    let mut rows = Vec::new();
    let node_sets: Vec<_> = pattern
        .nodes
        .iter()
        .map(|n| {
            n.filter.as_ref().map_or_else(
                || view.nodes().clone(),
                |f| view.filter_nodes(f).unwrap().nodes().clone(),
            )
        })
        .collect();
    let edge_sets: Vec<_> = pattern
        .edges
        .iter()
        .map(|e| {
            e.filter.as_ref().map_or_else(
                || view.edges().clone(),
                |f| view.filter_edges(f).unwrap().edges().clone(),
            )
        })
        .collect();
    let positions: HashMap<_, _> = pattern
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.variable.as_str(), i))
        .collect();
    for ni in 0..nodes.len().pow(pattern.nodes.len() as u32) {
        let ns = decode(ni, &nodes, pattern.nodes.len());
        if options.distinct_nodes && ns.iter().collect::<HashSet<_>>().len() != ns.len() {
            continue;
        }
        if ns
            .iter()
            .zip(&node_sets)
            .any(|(id, allowed)| !allowed.contains(id))
        {
            continue;
        }
        for ei in 0..edges.len().pow(pattern.edges.len() as u32) {
            let es = decode(ei, &edges, pattern.edges.len());
            if options.distinct_edges && es.iter().collect::<HashSet<_>>().len() != es.len() {
                continue;
            }
            let valid = pattern.edges.iter().enumerate().all(|(i, e)| {
                if !edge_sets[i].contains(&es[i]) {
                    return false;
                }
                let pair = (
                    ns[positions[e.source.as_str()]],
                    ns[positions[e.target.as_str()]],
                );
                let actual = view.edge_endpoints(es[i]).unwrap();
                let reverse = (actual.1, actual.0);
                if !view.is_directed() || e.direction == D::Both {
                    pair == actual || pair == reverse
                } else if e.direction == D::Incoming {
                    pair == reverse
                } else {
                    pair == actual
                }
            });
            if valid {
                rows.push(PatternMatch {
                    nodes: ns.clone(),
                    edges: es,
                });
            }
        }
    }
    rows.sort_by(|a, b| a.nodes.cmp(&b.nodes).then(a.edges.cmp(&b.edges)));
    rows
}
#[test]
fn graph_patterns_match_cartesian_reference_on_every_small_topology_and_direction() {
    for directed in [false, true] {
        for mask in 0..512 {
            let mut graph = if directed {
                Graph::new_directed()
            } else {
                Graph::new()
            };
            let mut edges = Vec::new();
            for a in 0..3 {
                for b in 0..3 {
                    if mask & (1 << (a * 3 + b)) != 0 {
                        edges.push((N::Added(a), N::Added(b)));
                    }
                }
            }
            if mask & 1 != 0 {
                edges.push((N::Added(0), N::Added(0)));
            }
            graph
                .apply_batch(
                    GraphBatch {
                        add_nodes: 3,
                        add_edges: edges,
                        ..Default::default()
                    },
                    0,
                    "fixture".into(),
                    "test".into(),
                )
                .unwrap();
            let view = graph.read_view().unwrap();
            for scope in [
                view.clone(),
                view.select_edges(&view.edge_ids().into_iter().step_by(2).collect::<Vec<_>>())
                    .unwrap(),
            ] {
                let index = scope.build_index(&[], &[], true).unwrap();
                for direction in [D::Outgoing, D::Incoming, D::Both] {
                    for distinct in [false, true] {
                        let options = PatternOptions {
                            distinct_nodes: distinct,
                            distinct_edges: distinct,
                            ..Default::default()
                        };
                        let pattern = path(direction);
                        let expected = brute(&scope, &pattern, &options);
                        let result = index.match_pattern(&pattern, &options, false).unwrap();
                        assert_eq!(result.rows(),expected,"directed={directed} mask={mask} direction={direction:?} distinct={distinct}");
                        assert!(!result.truncated());
                    }
                }
            }
        }
    }
}
fn fixture() -> Graph {
    let mut g = Graph::new_directed();
    g.insert_nodes(4).unwrap();
    for (a, b) in [(0, 1), (0, 1), (1, 2), (2, 0), (1, 1), (0, 2)] {
        g.add_edge(a, b).unwrap();
    }
    g.set_node_attr(0, "kind".into(), V::Text("anchor".into()))
        .unwrap();
    g.set_edge_attr(0, "weight".into(), V::Int(2)).unwrap();
    g
}
#[test]
fn branch_cycle_self_loop_disconnected_and_filtered_patterns_preserve_binding_multiplicity() {
    let g = fixture();
    let view = g.read_view().unwrap();
    let index = view
        .build_index(&["kind".into()], &["weight".into()], true)
        .unwrap();
    let mut patterns = vec![
        GraphPattern {
            nodes: vec![node("a"), node("b"), node("c")],
            edges: vec![
                edge("ab", "a", "b", D::Outgoing),
                edge("ac", "a", "c", D::Both),
            ],
        },
        GraphPattern {
            nodes: vec![node("a"), node("b"), node("c")],
            edges: vec![
                edge("ab", "a", "b", D::Outgoing),
                edge("bc", "b", "c", D::Outgoing),
                edge("ca", "c", "a", D::Outgoing),
            ],
        },
        GraphPattern {
            nodes: vec![node("a")],
            edges: vec![edge("loop", "a", "a", D::Both)],
        },
        GraphPattern {
            nodes: vec![node("a"), node("isolated")],
            edges: vec![],
        },
        GraphPattern {
            nodes: vec![node("a"), node("b")],
            edges: vec![
                edge("x", "a", "b", D::Outgoing),
                edge("y", "a", "b", D::Outgoing),
            ],
        },
    ];
    let mut filtered = path(D::Outgoing);
    filtered.nodes[0].filter = Some(NodeFilter::AttributeEquals {
        name: "kind".into(),
        value: V::Text("anchor".into()),
    });
    filtered.edges[0].filter = Some(EdgeFilter::HasAttribute {
        name: "weight".into(),
    });
    patterns.push(filtered);
    for pattern in patterns {
        for dn in [false, true] {
            for de in [false, true] {
                let options = PatternOptions {
                    distinct_nodes: dn,
                    distinct_edges: de,
                    ..Default::default()
                };
                let result = index.match_pattern(&pattern, &options, false).unwrap();
                assert_eq!(result.rows(), brute(&view, &pattern, &options));
                if !result.rows().is_empty() {
                    let selected = result.match_view(0).unwrap();
                    assert_eq!(
                        selected.nodes(),
                        &result.rows()[0].nodes.iter().copied().collect()
                    );
                    assert_eq!(
                        selected.edges(),
                        &result.rows()[0].edges.iter().copied().collect()
                    );
                }
            }
        }
    }
}
#[test]
fn limit_offset_and_expansion_budget_never_claim_unknown_completeness() {
    let g = fixture();
    let index = g.read_view().unwrap().build_index(&[], &[], true).unwrap();
    let pattern = path(D::Both);
    let all = index
        .match_pattern(&pattern, &PatternOptions::default(), false)
        .unwrap();
    for offset in [0, 1, 5, usize::MAX] {
        for limit in [Some(0), Some(1), Some(3), None] {
            let options = PatternOptions {
                offset,
                limit,
                ..Default::default()
            };
            let result = index.match_pattern(&pattern, &options, false).unwrap();
            let expected: Vec<_> = all
                .rows()
                .iter()
                .skip(offset)
                .take(limit.unwrap_or(usize::MAX))
                .cloned()
                .collect();
            assert_eq!(result.rows(), expected);
            assert_eq!(
                result.truncated(),
                limit.is_some_and(|n| all.rows().len().saturating_sub(offset) > n)
            );
        }
    }
    let required = all.expansions();
    assert!(index
        .match_pattern(
            &pattern,
            &PatternOptions {
                max_expansions: Some(required - 1),
                ..Default::default()
            },
            false
        )
        .is_err());
    assert_eq!(
        index
            .match_pattern(
                &pattern,
                &PatternOptions {
                    max_expansions: Some(required),
                    ..Default::default()
                },
                false
            )
            .unwrap()
            .rows(),
        all.rows()
    );
    assert!(index
        .match_pattern(
            &pattern,
            &PatternOptions {
                max_expansions: Some(0),
                ..Default::default()
            },
            false
        )
        .is_err());
    assert!(all.match_view(all.rows().len()).is_err());
}
#[test]
fn validation_and_index_availability_precede_empty_search_and_scans_are_explicit() {
    let g = fixture();
    let view = g.read_view().unwrap();
    let index = view.build_index(&[], &[], false).unwrap();
    assert!(index
        .match_pattern(&path(D::Outgoing), &PatternOptions::default(), false)
        .is_err());
    let scan = index
        .match_pattern(&path(D::Outgoing), &PatternOptions::default(), true)
        .unwrap();
    assert_eq!(
        scan.rows(),
        brute(&view, &path(D::Outgoing), &PatternOptions::default())
    );
    assert!(!index.has_topology());
    for pattern in [
        GraphPattern::default(),
        GraphPattern {
            nodes: vec![node("a"), node("a")],
            edges: vec![],
        },
        GraphPattern {
            nodes: vec![node("a")],
            edges: vec![edge("a", "a", "a", D::Both)],
        },
        GraphPattern {
            nodes: vec![node("a")],
            edges: vec![edge("x", "a", "unknown", D::Both)],
        },
    ] {
        assert!(pattern.validate().is_err());
        assert!(index
            .match_pattern(&pattern, &PatternOptions::default(), true)
            .is_err());
    }
    let indexed = view.build_index(&[], &[], true).unwrap();
    let mut pattern = path(D::Outgoing);
    pattern.nodes[0].filter = Some(NodeFilter::NodeSet(Default::default()));
    pattern.edges[1].filter = Some(EdgeFilter::HasAttribute {
        name: "missing".into(),
    });
    assert!(indexed
        .match_pattern(&pattern, &PatternOptions::default(), false)
        .is_err());
    assert!(indexed
        .match_pattern(&pattern, &PatternOptions::default(), true)
        .unwrap()
        .rows()
        .is_empty());
}
#[test]
fn result_rows_keep_exact_capture_clocks_isolates_and_owner_independent_lifetimes() {
    let mut g = Graph::new_directed();
    g.apply_batch(
        GraphBatch {
            add_nodes: 3,
            add_edges: vec![(N::Added(0), N::Added(1))],
            effective_at: Some(10),
            ..Default::default()
        },
        0,
        "initial".into(),
        "test".into(),
    )
    .unwrap();
    let view = g.read_view_at(15, None).unwrap();
    let index = view.build_index(&[], &[], true).unwrap();
    let pattern = GraphPattern {
        nodes: vec![PatternNode {
            variable: "alone".into(),
            filter: Some(NodeFilter::NodeSet([2].into_iter().collect())),
        }],
        edges: vec![],
    };
    let head = g.current_state_id();
    let result = index
        .match_pattern(&pattern, &PatternOptions::default(), false)
        .unwrap();
    assert_eq!(g.current_state_id(), head);
    assert_eq!(result.match_view(0).unwrap().node_ids(), vec![2]);
    assert!(result.match_view(0).unwrap().edge_ids().is_empty());
    assert_eq!(
        result.match_view(0).unwrap().read_state().read_time(),
        view.read_state().read_time()
    );
    g.remove_node(2).unwrap();
    drop(g);
    drop(pattern);
    drop(index);
    std::thread::spawn(move || {
        assert_eq!(result.match_view(0).unwrap().node_ids(), vec![2]);
        assert!(result.match_view(0).unwrap().write_state_id().is_none());
    })
    .join()
    .unwrap();
}
#[test]
fn deep_patterns_use_iterative_frames_and_do_not_require_recursion() {
    let mut g = Graph::new();
    g.insert_nodes(1).unwrap();
    g.add_edge(0, 0).unwrap();
    let count = 2000;
    let nodes = (0..count).map(|n| node(&format!("n{n}"))).collect();
    let edges = (1..count)
        .map(|n| {
            edge(
                &format!("e{n}"),
                &format!("n{}", n - 1),
                &format!("n{n}"),
                D::Outgoing,
            )
        })
        .collect();
    let result = g
        .read_view()
        .unwrap()
        .build_index(&[], &[], true)
        .unwrap()
        .match_pattern(
            &GraphPattern { nodes, edges },
            &PatternOptions {
                limit: Some(1),
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert!(!result.truncated());
    assert_eq!(result.rows()[0].nodes.len(), count);
}
