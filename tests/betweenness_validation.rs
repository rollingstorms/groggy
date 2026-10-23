use groggy::algorithms::centrality::BetweennessCentrality;
use groggy::algorithms::{Algorithm, AlgorithmOutput, Context};
use groggy::subgraphs::Subgraph;
use groggy::{AttrValue, Graph};
use std::{cell::RefCell, rc::Rc};

type SharedGraph = Rc<RefCell<Graph>>;
fn selected(g: &SharedGraph) -> Subgraph {
    Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "betweenness".into()).unwrap()
}
fn execute(g: &SharedGraph, weighted: bool, normalized: bool) -> anyhow::Result<()> {
    BetweennessCentrality::new(normalized, weighted.then(|| "w".into()), "score".into())
        .execute(&mut Context::new(), selected(g))?;
    Ok(())
}
fn score(g: &SharedGraph, n: usize) -> f64 {
    match g
        .borrow()
        .get_node_attr(n, &"score".into())
        .unwrap()
        .unwrap()
    {
        AttrValue::Float(v) => v as f64,
        other => panic!("unexpected {other:?}"),
    }
}

// Enumerate simple edge-instance paths independently of Brandes/CSR.
// Strictly positive lengths ensure a shortest path is simple.
fn reference(
    n: usize,
    edges: &[(usize, usize, f64)],
    directed: bool,
    normalized: bool,
) -> Vec<f64> {
    fn visit(
        here: usize,
        target_and_direction: (usize, bool),
        edges: &[(usize, usize, f64)],
        path: &mut Vec<usize>,
        cost: f64,
        best: &mut f64,
        paths: &mut Vec<Vec<usize>>,
    ) {
        let (target, directed) = target_and_direction;
        if here == target {
            if cost < *best {
                *best = cost;
                paths.clear();
            }
            if cost == *best {
                paths.push(path.clone());
            }
            return;
        }
        for &(u, v, w) in edges {
            let next = if u == here {
                v
            } else if !directed && v == here {
                u
            } else {
                continue;
            };
            if path.contains(&next) {
                continue;
            }
            path.push(next);
            visit(next, (target, directed), edges, path, cost + w, best, paths);
            path.pop();
        }
    }
    let mut scores = vec![0.; n];
    for source in 0..n {
        for target in 0..n {
            if source == target {
                continue;
            }
            let mut paths = Vec::new();
            let mut best = f64::INFINITY;
            visit(
                source,
                (target, directed),
                edges,
                &mut vec![source],
                0.,
                &mut best,
                &mut paths,
            );
            for path in &paths {
                for &node in &path[1..path.len() - 1] {
                    scores[node] += 1. / paths.len() as f64;
                }
            }
        }
    }
    for s in &mut scores {
        if normalized && n > 2 {
            *s /= ((n - 1) * (n - 2)) as f64;
        } else if !directed {
            *s /= 2.;
        }
    }
    scores
}

#[test]
fn directed_and_undirected_scores_match_exhaustive_edge_path_reference() {
    let edges = [
        (1, 0, 1.),
        (1, 0, 1.),
        (1, 0, 7.),
        (0, 2, 1.),
        (1, 3, 1.),
        (3, 2, 1.),
        (2, 4, 2.),
        (4, 1, 3.),
        (2, 2, 1.),
    ];
    for directed in [false, true] {
        let mut graph = if directed {
            Graph::new_directed()
        } else {
            Graph::new()
        };
        graph.add_nodes(6); // Include an isolate in normalization.
        for (i, &(u, v, w)) in edges.iter().enumerate() {
            let e = graph.add_edge(u, v).unwrap();
            if i != 4 {
                graph
                    .set_edge_attr(e, "w".into(), AttrValue::Int(w as i64))
                    .unwrap();
            }
        }
        let g = Rc::new(RefCell::new(graph));
        // Warm the shared unweighted CSR, then verify weighted filtering doesn't contaminate it.
        for (weighted, normalized) in [(false, false), (true, false), (true, true), (false, true)] {
            execute(&g, weighted, normalized).unwrap();
            let reference_edges: Vec<_> = edges
                .iter()
                .map(|&(u, v, w)| (u, v, if weighted { w } else { 1. }))
                .collect();
            let expected = reference(6, &reference_edges, directed, normalized);
            let frozen = g
                .borrow()
                .read_view()
                .unwrap()
                .betweenness(normalized, weighted.then(|| "w".into()))
                .unwrap();
            let mut ctx = Context::default();
            ctx.set_persist_results(false);
            BetweennessCentrality::new(
                normalized,
                weighted.then(|| "w".into()),
                "not_written".into(),
            )
            .execute(&mut ctx, selected(&g))
            .unwrap();
            let AlgorithmOutput::NodeScores(transient) = ctx
                .take_outputs()
                .remove("centrality.betweenness.scores")
                .unwrap()
            else {
                panic!("expected scores")
            };
            assert_eq!(transient.len(), expected.len());
            for (rows, want) in [frozen, transient]
                .iter()
                .flat_map(|rows| rows.iter().zip(&expected))
            {
                assert!((rows.1 - want).abs() < 1e-12);
                assert!(g
                    .borrow()
                    .get_node_attr(rows.0, &"not_written".into())
                    .unwrap()
                    .is_none());
            }
            for (node, want) in expected.iter().enumerate() {
                assert!((score(&g,node)-want).abs() < 1e-6, "directed={directed} weighted={weighted} normalized={normalized} node={node}: {} != {want}",score(&g,node));
            }
        }
    }
}

#[test]
fn zero_invalid_and_nonfinite_weights_preserve_prior_results() {
    for bad in [
        AttrValue::Int(0),
        AttrValue::Int(-1),
        AttrValue::Null,
        AttrValue::Bool(true),
        AttrValue::Text("bad".into()),
        AttrValue::Float(f32::NAN),
        AttrValue::Float(f32::INFINITY),
    ] {
        let mut graph = Graph::new();
        graph.add_nodes(3);
        let e = graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();
        let g = Rc::new(RefCell::new(graph));
        execute(&g, true, false).unwrap();
        let before: Vec<_> = (0..3).map(|n| score(&g, n)).collect();
        g.borrow_mut().set_edge_attr(e, "w".into(), bad).unwrap();
        assert!(execute(&g, true, false).is_err());
        assert_eq!((0..3).map(|n| score(&g, n)).collect::<Vec<_>>(), before);
        assert!(execute(&g, false, false).is_ok()); // Unit mode ignores unused property values.
    }
}

#[test]
fn tiny_positive_lengths_do_not_become_spurious_ties() {
    let edges = [(0, 1, 4e-20_f32), (0, 2, 1e-20), (2, 1, 1e-20)];
    let mut graph = Graph::new_directed();
    graph.add_nodes(3);
    for &(u, v, w) in &edges {
        let e = graph.add_edge(u, v).unwrap();
        graph
            .set_edge_attr(e, "w".into(), AttrValue::Float(w))
            .unwrap();
    }
    let g = Rc::new(RefCell::new(graph));
    execute(&g, true, false).unwrap();
    assert_eq!(score(&g, 2), 1.);
}

#[test]
fn stale_selection_rejects_even_for_a_single_node() {
    let mut graph = Graph::new();
    let node = graph.add_node();
    let g = Rc::new(RefCell::new(graph));
    let sub = selected(&g);
    g.borrow_mut().remove_node(node).unwrap();
    assert!(BetweennessCentrality::new(false, None, "score".into())
        .execute(&mut Context::new(), sub)
        .is_err());
}

#[test]
fn rounded_away_distance_and_path_count_overflow_reject() {
    let mut graph = Graph::new_directed();
    graph.add_nodes(3);
    for (u, v, w) in [(0, 1, 1e20_f32), (1, 2, 1.)] {
        let e = graph.add_edge(u, v).unwrap();
        graph
            .set_edge_attr(e, "w".into(), AttrValue::Float(w))
            .unwrap();
    }
    assert!(execute(&Rc::new(RefCell::new(graph)), true, false)
        .unwrap_err()
        .to_string()
        .contains("cannot advance"));
    let mut graph = Graph::new_directed();
    graph.add_nodes(1030);
    for n in 0..1029 {
        graph.add_edge(n, n + 1).unwrap();
        graph.add_edge(n, n + 1).unwrap();
    }
    let g = Rc::new(RefCell::new(graph));
    for weighted in [false, true] {
        assert!(execute(&g, weighted, false)
            .unwrap_err()
            .to_string()
            .contains("count exceeds"));
        assert!(g
            .borrow()
            .get_node_attr(0, &"score".into())
            .unwrap()
            .is_none());
    }
}

#[test]
fn captured_small_multigraphs_match_independent_edge_path_enumeration() {
    let pairs = [
        (0, 1),
        (0, 1),
        (1, 2),
        (0, 2),
        (2, 0),
        (1, 0),
        (2, 2),
        (2, 1),
    ];
    for directed in [false, true] {
        for mask in 0..256 {
            let mut graph = if directed {
                Graph::new_directed()
            } else {
                Graph::new()
            };
            graph.add_nodes(4); // Includes an isolate.
            let mut edges = Vec::new();
            for (bit, &(source, target)) in pairs.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    let edge = graph.add_edge(source, target).unwrap();
                    let weight = (bit % 3 + 1) as f64;
                    graph
                        .set_edge_attr(edge, "w".into(), AttrValue::Int(weight as i64))
                        .unwrap();
                    edges.push((source, target, weight));
                }
            }
            let view = graph.read_view().unwrap();
            for weighted in [false, true] {
                let reference_edges: Vec<_> = edges
                    .iter()
                    .map(|&(s, t, w)| (s, t, if weighted { w } else { 1. }))
                    .collect();
                for normalized in [false, true] {
                    let expected = reference(4, &reference_edges, directed, normalized);
                    let result = view
                        .betweenness(normalized, weighted.then(|| "w".into()))
                        .unwrap();
                    for ((node, actual), want) in result.iter().zip(expected) {
                        assert!((actual-want).abs()<1e-12,"mask={mask}, directed={directed}, weighted={weighted}, normalized={normalized}, node={node}");
                    }
                }
            }
        }
    }
}
