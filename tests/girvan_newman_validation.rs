use groggy::algorithms::community::GirvanNewman;
use groggy::algorithms::{
    ensure_algorithms_registered, global_registry, Algorithm, Context, PipelineBuilder,
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
fn triangles(reverse: bool, bridges: usize) -> Shared {
    let mut g = Graph::new();
    g.add_nodes(7);
    for (u, v) in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
        let (u, v) = if reverse { (v, u) } else { (u, v) };
        g.add_edge(u, v).unwrap();
    }
    for i in 0..bridges {
        let e = g
            .add_edge(if reverse { 3 } else { 2 }, if reverse { 2 } else { 3 })
            .unwrap();
        g.set_edge_attr(e, "w".into(), AttrValue::Int(if i == 0 { 1 } else { 100 }))
            .unwrap();
    }
    Rc::new(RefCell::new(g))
}
fn run(g: &Shared, levels: usize, weighted: bool) -> Context {
    let mut ctx = Context::new();
    GirvanNewman::new(Some(levels), None, weighted.then(|| "w".into()), "c".into())
        .execute(&mut ctx, selected(g))
        .unwrap();
    ctx
}
fn labels(g: &Shared) -> Vec<AttrValue> {
    (0..7)
        .map(|n| g.borrow().get_node_attr(n, &"c".into()).unwrap().unwrap())
        .collect()
}
#[test]
fn last_allowed_removal_is_scored_against_original_graph_in_both_orientations() {
    for reverse in [false, true] {
        let g = triangles(reverse, 1);
        let ctx = run(&g, 1, false);
        let c = labels(&g);
        assert_eq!(c[0], c[1]);
        assert_eq!(c[1], c[2]);
        assert_eq!(c[3], c[4]);
        assert_eq!(c[4], c[5]);
        assert_ne!(c[0], c[3]);
        assert_ne!(c[6], c[0]);
        assert_eq!(ctx.stat_snapshot()["girvan_newman.count.iterations"], 1.);
        assert!((ctx.stat_snapshot()["girvan_newman.best_modularity"] - 5. / 14.).abs() < 1e-12);
        assert_eq!(g.borrow().edge_ids().len(), 7); // Algorithm removals are local only.
    }
}
#[test]
fn parallel_bridges_require_separate_removals_in_weighted_and_unit_modes() {
    for weighted in [false, true] {
        let g = triangles(true, 2);
        run(&g, 1, weighted);
        let c = labels(&g);
        assert_eq!(c[0], c[3]);
        let ctx = run(&g, 2, weighted);
        let c = labels(&g);
        assert_ne!(c[0], c[3]);
        assert_eq!(ctx.stat_snapshot()["girvan_newman.count.iterations"], 2.);
        assert_eq!(ctx.stat_snapshot()["girvan_newman.best_modularity"], 0.25);
    }
}
#[test]
fn zero_budget_keeps_initial_components_and_loops_do_not_connect_isolates() {
    let g = triangles(false, 0);
    g.borrow_mut().add_edge(6, 6).unwrap();
    let ctx = run(&g, 0, false);
    let c = labels(&g);
    assert_eq!(c[0], c[2]);
    assert_ne!(c[0], c[3]);
    assert_ne!(c[0], c[6]);
    assert_eq!(ctx.stat_snapshot()["girvan_newman.count.iterations"], 0.);
}
#[test]
fn invalid_inputs_and_cancelled_or_stale_selections_preserve_existing_output() {
    let g = triangles(false, 1);
    run(&g, 1, true);
    let before = labels(&g);
    for bad in [
        AttrValue::Int(0),
        AttrValue::Int(-1),
        AttrValue::Null,
        AttrValue::Bool(true),
        AttrValue::Text("bad".into()),
        AttrValue::Float(f32::NAN),
        AttrValue::Float(f32::INFINITY),
    ] {
        g.borrow_mut().set_edge_attr(6, "w".into(), bad).unwrap();
        assert!(
            GirvanNewman::new(Some(0), None, Some("w".into()), "c".into())
                .execute(&mut Context::new(), selected(&g))
                .is_err()
        );
        assert_eq!(labels(&g), before);
    }
    for threshold in [-1., f64::NAN, f64::INFINITY] {
        assert!(
            GirvanNewman::new(Some(0), Some(threshold), None, "c".into())
                .execute(&mut Context::new(), selected(&g))
                .is_err()
        );
    }
    let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
    assert!(GirvanNewman::new(Some(0), None, None, "c".into())
        .execute(&mut ctx, selected(&g))
        .is_err());
    let sub = selected(&g);
    g.borrow_mut().remove_node(6).unwrap();
    assert!(GirvanNewman::new(Some(0), None, None, "c".into())
        .execute(&mut Context::new(), sub)
        .is_err());
    let mut directed = Graph::new_directed();
    directed.add_node();
    let g = Rc::new(RefCell::new(directed));
    assert!(GirvanNewman::new(None, None, None, "c".into())
        .execute(&mut Context::new(), selected(&g))
        .unwrap_err()
        .to_string()
        .contains("undirected"));
}
#[test]
fn negative_registry_level_budget_is_rejected() {
    ensure_algorithms_registered();
    assert!(PipelineBuilder::new()
        .with_algorithm("community.girvan_newman", |p| {
            p.set_int("num_levels", -1);
        })
        .build(global_registry())
        .is_err());
}

#[test]
fn empty_and_single_node_selections_are_valid() {
    for count in [0, 1] {
        let mut graph = Graph::new();
        graph.add_nodes(count);
        let g = Rc::new(RefCell::new(graph));
        GirvanNewman::new(None, None, None, "c".into())
            .execute(&mut Context::new(), selected(&g))
            .unwrap();
        if count == 1 {
            assert_eq!(
                g.borrow().get_node_attr(0, &"c".into()).unwrap(),
                Some(AttrValue::Int(0))
            );
        }
    }
}
