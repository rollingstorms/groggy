use groggy::algorithms::community::Leiden;
use groggy::algorithms::{Algorithm, Context};
use groggy::subgraphs::Subgraph;
use groggy::Graph;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

type Shared = Rc<RefCell<Graph>>;
fn ring() -> Shared {
    let mut g = Graph::new();
    g.add_nodes(50);
    g.remove_node(48).unwrap();
    for c in 0..16 {
        let a = c * 3;
        for (u, v) in [
            (a, a + 1),
            (a + 1, a + 2),
            (a + 2, a),
            (a + 2, (a + 3) % 48),
        ] {
            g.add_edge(u, v).unwrap();
        }
    }
    Rc::new(RefCell::new(g))
}
fn sub(g: &Shared) -> Subgraph {
    Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "test".into()).unwrap()
}
fn labels(g: &Shared) -> HashMap<usize, usize> {
    g.borrow()
        .node_ids()
        .into_iter()
        .map(|n| {
            (
                n,
                g.borrow()
                    .get_node_attr(n, &"c".into())
                    .unwrap()
                    .unwrap()
                    .as_int()
                    .unwrap() as usize,
            )
        })
        .collect()
}
fn run(g: &Shared, levels: usize, visits: usize, seed: u64) -> (HashMap<usize, usize>, Context) {
    let mut ctx = Context::new();
    Leiden::new(visits, levels, 1., Some(seed), "c".into())
        .unwrap()
        .execute(&mut ctx, sub(g))
        .unwrap();
    (labels(g), ctx)
}
// Original adjacency-matrix objective, independently of quotient scoring.
fn objective(g: &Shared, labels: &HashMap<usize, usize>) -> f64 {
    let g = g.borrow();
    let mut nodes = g.node_ids();
    nodes.sort_unstable();
    let index: HashMap<_, _> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut a = vec![vec![0.; nodes.len()]; nodes.len()];
    let mut degrees = vec![0.; nodes.len()];
    for e in g.edge_ids() {
        let (u, v) = g.edge_endpoints(e).unwrap();
        let (i, j) = (index[&u], index[&v]);
        a[i][j] += 1.;
        a[j][i] += 1.;
        degrees[i] += 1.;
        degrees[j] += 1.;
    }
    let twice_m = (2 * g.edge_ids().len()) as f64;
    let mut q = 0.;
    for (i, &u) in nodes.iter().enumerate() {
        for (j, &v) in nodes.iter().enumerate() {
            if labels[&u] == labels[&v] {
                q += a[i][j] - degrees[i] * degrees[j] / twice_m;
            }
        }
    }
    q / twice_m
}
fn assert_connected(g: &Shared, labels: &HashMap<usize, usize>) {
    let g = g.borrow();
    let edges: Vec<_> = g
        .edge_ids()
        .iter()
        .map(|&e| g.edge_endpoints(e).unwrap())
        .collect();
    for (&root, &community) in labels {
        let mut seen = HashSet::from([root]);
        let mut stack = vec![root];
        while let Some(u) = stack.pop() {
            for &(a, b) in &edges {
                let v = if a == u {
                    b
                } else if b == u {
                    a
                } else {
                    continue;
                };
                if labels[&v] == community && seen.insert(v) {
                    stack.push(v);
                }
            }
        }
        assert!(labels
            .iter()
            .all(|(&n, &c)| c != community || seen.contains(&n)));
    }
}
#[test]
fn refined_levels_improve_original_objective_and_preserve_id_projection() {
    let g = ring();
    let (single, _) = run(&g, 1, 100, 42);
    let (multi, ctx) = run(&g, 10, 100, 42);
    assert!(objective(&g, &multi) > objective(&g, &single) + 1e-6);
    assert!(ctx.stat_snapshot()["leiden.total_phases"] > 1.);
    assert!(multi.iter().all(|(&n, &c)| n == 49 || c != multi[&49]));
    assert_connected(&g, &multi);
    let stats = ctx.stat_snapshot();
    let mut previous = f64::NEG_INFINITY;
    for phase in 0..stats["leiden.total_phases"] as usize {
        let q = stats[&format!("leiden.phase_{phase}.modularity")];
        assert!(q >= previous - 1e-12);
        previous = q;
        assert!(
            stats[&format!("leiden.phase_{phase}.visits")]
                <= 100. * stats[&format!("leiden.phase_{phase}.nodes")]
        );
    }
    assert!((previous - objective(&g, &multi)).abs() < 1e-12);
    assert_eq!(run(&g, 10, 100, 42).0, multi);
    assert_eq!(g.borrow().edge_ids().len(), 64);
    assert!(!g.borrow().has_node(48));
}
#[test]
fn bounded_work_still_returns_connected_communities_with_loops_and_parallel_edges() {
    for seed in 0..12 {
        let g = ring();
        g.borrow_mut().add_edge(0, 0).unwrap();
        g.borrow_mut().add_edge(0, 1).unwrap();
        let (c, ctx) = run(&g, 3, 1, seed);
        assert_connected(&g, &c);
        let stats = ctx.stat_snapshot();
        for phase in 0..stats["leiden.total_phases"] as usize {
            assert!(
                stats[&format!("leiden.phase_{phase}.visits")]
                    <= stats[&format!("leiden.phase_{phase}.nodes")]
            );
        }
        assert_eq!(g.borrow().edge_ids().len(), 66);
    }
}
#[test]
fn invalid_theta_and_visit_overflow_do_not_replace_published_labels() {
    let g = ring();
    let (before, _) = run(&g, 2, 100, 3);
    for theta in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(Leiden::new(20, 10, 1., Some(42), "c".into())
            .unwrap()
            .with_theta(theta)
            .is_err());
    }
    let error = Leiden::new(usize::MAX, 10, 1., Some(42), "c".into())
        .unwrap()
        .execute(&mut Context::new(), sub(&g))
        .unwrap_err();
    assert!(error.to_string().contains("budget overflow"));
    assert_eq!(labels(&g), before);
}
#[test]
fn explicit_no_persistence_computes_without_writing_labels() {
    let g = ring();
    let mut ctx = Context::new();
    ctx.set_persist_results(false);
    Leiden::new(100, 10, 1., Some(42), "c".into())
        .unwrap()
        .execute(&mut ctx, sub(&g))
        .unwrap();
    assert!(ctx.stat_snapshot()["leiden.total_phases"] > 1.);
    assert_eq!(g.borrow().get_node_attr(0, &"c".into()).unwrap(), None);
}
