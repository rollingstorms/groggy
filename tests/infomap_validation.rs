use groggy::algorithms::community::Infomap;
use groggy::algorithms::{
    ensure_algorithms_registered, global_registry, Algorithm, Context, PipelineBuilder,
};
use groggy::subgraphs::Subgraph;
use groggy::Graph;
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{atomic::AtomicBool, Arc},
};
type Shared = Rc<RefCell<Graph>>;
fn selected(g: &Shared) -> Subgraph {
    Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "test".into()).unwrap()
}
fn cliques() -> Shared {
    let mut graph = Graph::new_manual();
    graph.add_nodes(10).unwrap();
    graph.remove_node(8).unwrap();
    graph.remove_node(9).unwrap();
    for start in [0, 4] {
        for u in start..start + 4 {
            for v in u + 1..start + 4 {
                graph.add_edge(u, v).unwrap();
            }
        }
    }
    graph.add_edge(3, 4).unwrap();
    Rc::new(RefCell::new(graph))
}
fn run(g: &Shared, tau: f64, trials: usize) -> Context {
    let mut ctx = Context::new();
    Infomap::new(tau, trials, 50, Some(42), "c".into())
        .unwrap()
        .execute(&mut ctx, selected(g))
        .unwrap();
    ctx
}
fn labels(g: &Shared) -> Vec<i64> {
    let mut ids = g.borrow().node_ids();
    ids.sort_unstable();
    ids.iter()
        .map(|&n| {
            g.borrow()
                .get_node_attr(n, &"c".into())
                .unwrap()
                .unwrap()
                .as_int()
                .unwrap()
        })
        .collect()
}
#[test]
fn real_map_equation_finds_flow_modules_and_uses_teleportation_and_all_trials() {
    let g = cliques();
    let ctx = run(&g, 0.15, 5);
    let c = labels(&g);
    assert!(c[..4].iter().all(|&v| v == c[0]));
    assert!(c[4..].iter().all(|&v| v == c[4]));
    assert_ne!(c[0], c[4]);
    assert!(
        ctx.stat_snapshot()["infomap.best_bits"] < ctx.stat_snapshot()["infomap.one_module_bits"]
    );
    assert_eq!(ctx.stat_snapshot()["infomap.trials"], 5.);
    for trial in 0..5 {
        assert!(ctx
            .stat_snapshot()
            .contains_key(&format!("infomap.trial_{trial}.bits")));
    }
    let initial = ctx.stat_snapshot()["infomap.best_bits"];
    assert_eq!(
        run(&g, 0.15, 5).stat_snapshot()["infomap.best_bits"],
        initial
    );
    run(&g, 1., 5);
    let all = labels(&g);
    assert!(all.iter().all(|&v| v == all[0]));
    assert_eq!(g.borrow().edge_ids().len(), 13);
}
#[test]
fn direction_changes_stationary_code_length_and_rng_is_local() {
    for directed in [false, true] {
        let mut graph = if directed {
            Graph::new_directed_manual()
        } else {
            Graph::new_manual()
        };
        graph.add_nodes(2).unwrap();
        graph.add_edge(0, 1).unwrap();
        let g = Rc::new(RefCell::new(graph));
        fastrand::seed(123);
        let expected = fastrand::u64(..);
        fastrand::seed(123);
        let ctx = run(&g, 0.25, 3);
        assert_eq!(fastrand::u64(..), expected);
        let pi = if directed { 4. / 11. } else { 0.5 };
        let want = -pi * f64::log2(pi) - (1. - pi) * f64::log2(1. - pi);
        assert!((ctx.stat_snapshot()["infomap.best_bits"] - want).abs() < 1e-10);
        assert!(ctx.stat_snapshot()["infomap.flow.residual"] <= 1e-12);
    }
}
#[test]
fn failed_flow_stale_selection_and_cancellation_preserve_previous_labels() {
    let g = cliques();
    run(&g, 0.15, 1);
    let before = labels(&g);
    let error = Infomap::new(0.15, 1, 10, Some(42), "c".into())
        .unwrap()
        .with_flow_budget(1)
        .unwrap()
        .execute(&mut Context::new(), selected(&g))
        .unwrap_err();
    assert!(error.to_string().contains("did not converge"));
    assert_eq!(labels(&g), before);
    let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
    assert!(Infomap::new(0.15, 1, 10, None, "c".into())
        .unwrap()
        .execute(&mut ctx, selected(&g))
        .is_err());
    assert_eq!(labels(&g), before);
    let sub = selected(&g);
    g.borrow_mut().remove_node(0).unwrap();
    assert!(Infomap::new(0.15, 1, 10, None, "c".into())
        .unwrap()
        .execute(&mut Context::new(), sub)
        .is_err());
}
#[test]
fn empty_singleton_edgeless_and_id_gap_cases_are_defined() {
    for n in [0, 1, 3] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(n).unwrap();
        let g = Rc::new(RefCell::new(graph));
        run(&g, 0.15, 2);
        assert_eq!(labels(&g), vec![0; n]);
    }
    let mut graph = Graph::new_directed_manual();
    graph.add_nodes(4).unwrap();
    graph.remove_node(1).unwrap();
    graph.add_edge(0, 2).unwrap();
    graph.add_edge(2, 3).unwrap();
    graph.add_edge(3, 0).unwrap();
    let g = Rc::new(RefCell::new(graph));
    run(&g, 0., 2);
    assert_eq!(labels(&g).len(), 3);
    assert!(g.borrow().get_node_attr(1, &"c".into()).is_err());
}
#[test]
fn invalid_controls_reject_in_rust_and_registry() {
    for tau in [-1., f64::NAN, f64::INFINITY, 1.1] {
        assert!(Infomap::new(tau, 1, 10, None, "c".into()).is_err());
    }
    assert!(Infomap::new(0.15, 1, 10, None, "c".into())
        .unwrap()
        .with_flow_budget(0)
        .is_err());
    ensure_algorithms_registered();
    for param in ["num_trials", "max_iter", "flow_max_iter"] {
        for value in [-1, 0] {
            assert!(PipelineBuilder::new()
                .with_algorithm("community.infomap", |p| p.set_int(param, value))
                .build(global_registry())
                .is_err());
        }
    }
    assert!(PipelineBuilder::new()
        .with_algorithm("community.infomap", |p| p.set_int("seed", -1))
        .build(global_registry())
        .is_err());
}
