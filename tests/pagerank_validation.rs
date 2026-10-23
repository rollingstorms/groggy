use groggy::algorithms::centrality::PageRank;
use groggy::algorithms::{
    ensure_algorithms_registered, global_registry, Algorithm, AlgorithmOutput, Context,
    PipelineBuilder,
};
use groggy::subgraphs::Subgraph;
use groggy::{AttrValue, Graph};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{atomic::AtomicBool, Arc},
};
type Shared = Rc<RefCell<Graph>>;
fn selected(g: &Shared) -> Subgraph {
    Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "test".into()).unwrap()
}
fn run(g: &Shared, damping: f64, personal: bool, budget: usize) -> (Vec<(usize, f64)>, Context) {
    let mut ctx = Context::new();
    ctx.set_persist_results(false);
    PageRank::new(
        damping,
        budget,
        1e-13,
        personal.then(|| "p".into()),
        "pr".into(),
    )
    .unwrap()
    .execute(&mut ctx, selected(g))
    .unwrap();
    let AlgorithmOutput::NodeScores(scores) = ctx
        .take_outputs()
        .remove("centrality.pagerank.scores")
        .unwrap()
    else {
        panic!("expected scores")
    };
    (scores, ctx)
}

// Solve (I - alpha P^T)x = (1-alpha)p directly, without power iteration/CSR.
fn solve(p: [[f64; 3]; 3], alpha: f64, reset: [f64; 3]) -> [f64; 3] {
    let mut a = [[0.; 4]; 3];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, value) in row[..3].iter_mut().enumerate() {
            *value = f64::from(i == j) - alpha * p[j][i];
        }
        row[3] = (1. - alpha) * reset[i];
    }
    for k in 0..3 {
        let pivot = (k..3)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        a.swap(k, pivot);
        let scale = a[k][k];
        for v in &mut a[k][k..] {
            *v /= scale;
        }
        let normalized = a[k];
        for (i, row) in a.iter_mut().enumerate() {
            if i == k {
                continue;
            }
            let multiplier = row[k];
            for (j, value) in row.iter_mut().enumerate().skip(k) {
                *value -= multiplier * normalized[j];
            }
        }
    }
    [a[0][3], a[1][3], a[2][3]]
}

#[test]
fn all_three_node_graphs_match_a_direct_linear_solve() {
    for directed in [false, true] {
        for mask in 0..512 {
            let mut graph = if directed {
                Graph::new_directed()
            } else {
                Graph::new()
            };
            graph.add_nodes(3);
            let mut adjacency = [[0.; 3]; 3];
            for bit in 0..9 {
                let (u, v) = (bit / 3, bit % 3);
                if mask & (1 << bit) != 0 {
                    graph.add_edge(u, v).unwrap();
                    adjacency[u][v] += 1.;
                    if !directed && u != v {
                        adjacency[v][u] += 1.;
                    }
                }
            }
            for (node, value) in [1, 3, 0].into_iter().enumerate() {
                graph
                    .set_node_attr(node, "p".into(), AttrValue::Int(value))
                    .unwrap();
            }
            let g = Rc::new(RefCell::new(graph));
            for personal in [false, true] {
                let reset = if personal {
                    [0.25, 0.75, 0.]
                } else {
                    [1. / 3.; 3]
                };
                let mut p = adjacency;
                for row in &mut p {
                    let degree: f64 = row.iter().sum();
                    if degree == 0. {
                        *row = reset;
                    } else {
                        for value in row {
                            *value /= degree;
                        }
                    }
                }
                for alpha in [0., 0.85] {
                    let want = solve(p, alpha, reset);
                    let frozen = g
                        .borrow()
                        .read_view()
                        .unwrap()
                        .pagerank(alpha, 1000, 1e-13, personal.then(|| "p".into()))
                        .unwrap();
                    assert!(frozen.converged);
                    assert!(frozen.max_update <= 1e-13);
                    assert!(frozen.iterations > 0 && frozen.iterations <= 1000);
                    for ((_, actual), expected) in frozen.scores.iter().zip(want) {
                        assert!((actual - expected).abs() < 2e-12);
                    }
                    let (scores, ctx) = run(&g, alpha, personal, 1000);
                    assert_eq!(ctx.stat_snapshot()["pr.converged"], 1.);
                    assert!((scores.iter().map(|(_, s)| s).sum::<f64>() - 1.).abs() < 1e-14);
                    for ((node, got), expected) in scores.iter().zip(want) {
                        assert!((got-expected).abs() < 2e-12,"directed={directed}, mask={mask}, personal={personal}, alpha={alpha}, node={node}, got={got}, want={expected}");
                    }
                }
            }
        }
    }
}

#[test]
fn bad_personalization_and_stale_cancelled_runs_preserve_prior_scores() {
    for bad in [
        AttrValue::Float(f32::NAN),
        AttrValue::Float(f32::INFINITY),
        AttrValue::Int(-1),
        AttrValue::Bool(true),
        AttrValue::Text("bad".into()),
        AttrValue::Null,
    ] {
        let mut graph = Graph::new_directed();
        graph.add_nodes(2);
        graph.add_edge(0, 1).unwrap();
        graph.set_node_attr(0, "p".into(), bad).unwrap();
        graph
            .set_node_attr(0, "pr".into(), AttrValue::Int(7))
            .unwrap();
        let g = Rc::new(RefCell::new(graph));
        let algo = PageRank::new(0.85, 100, 1e-12, Some("p".into()), "pr".into()).unwrap();
        assert!(algo.execute(&mut Context::new(), selected(&g)).is_err());
        assert_eq!(
            g.borrow().get_node_attr(0, &"pr".into()).unwrap(),
            Some(AttrValue::Int(7))
        );
    }
    let mut graph = Graph::new();
    graph.add_nodes(2);
    graph.add_edge(0, 1).unwrap();
    let g = Rc::new(RefCell::new(graph));
    let sub = selected(&g);
    let algo = PageRank::new(0.85, 100, 1e-12, None, "pr".into()).unwrap();
    let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
    assert!(algo.execute(&mut ctx, sub.clone()).is_err());
    assert!(g.borrow().get_node_attr(0, &"pr".into()).unwrap().is_none());
    g.borrow_mut().remove_node(1).unwrap();
    assert!(algo.execute(&mut Context::new(), sub).is_err());
}

#[test]
fn empty_singleton_missing_personalization_sparse_ids_and_bounded_runs() {
    for n in [0, 1, 3] {
        let mut graph = Graph::new();
        graph.add_nodes(n);
        let g = Rc::new(RefCell::new(graph));
        let (scores, _) = run(&g, 0.85, false, 10);
        assert_eq!(scores.len(), n);
        for (_, value) in scores {
            assert!((value - 1. / n as f64).abs() < 1e-14);
        }
    }
    let mut graph = Graph::new_directed();
    graph.add_nodes(4);
    graph.remove_node(1).unwrap();
    graph.add_edge(0, 2).unwrap();
    graph.add_edge(0, 2).unwrap();
    graph.add_edge(0, 0).unwrap();
    // Missing personalization remains weight one, while compact zero is zero.
    graph
        .set_node_attr(3, "p".into(), AttrValue::Int(0))
        .unwrap();
    let g = Rc::new(RefCell::new(graph));
    let (scores, ctx) = run(&g, 0.85, true, 1);
    assert_eq!(ctx.stat_snapshot()["pr.converged"], 0.);
    assert_eq!(
        scores.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![0, 2, 3]
    );
    assert_eq!(scores[2].1, 0.);
    for id in [0, 2, 3] {
        g.borrow_mut()
            .set_node_attr(id, "p".into(), AttrValue::Int(0))
            .unwrap();
    }
    assert!(
        PageRank::new(0.85, 100, 1e-12, Some("p".into()), "pr".into())
            .unwrap()
            .execute(&mut Context::new(), selected(&g))
            .is_err()
    );
}

#[test]
fn invalid_controls_reject_even_for_empty_input() {
    for tol in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(PageRank::new(0.85, 10, tol, None, "pr".into()).is_err());
    }
    for damping in [-1., 1., f64::NAN, f64::INFINITY] {
        assert!(PageRank::new(damping, 10, 1e-6, None, "pr".into()).is_err());
    }
    ensure_algorithms_registered();
    for budget in [0, -1] {
        assert!(PipelineBuilder::new()
            .with_algorithm("centrality.pagerank", |p| p.set_int("max_iter", budget))
            .build(global_registry())
            .is_err());
    }
    let g = Rc::new(RefCell::new(Graph::new()));
    let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
    assert!(PageRank::new(0.85, 10, 1e-6, None, "pr".into())
        .unwrap()
        .execute(&mut ctx, selected(&g))
        .is_err());
}
