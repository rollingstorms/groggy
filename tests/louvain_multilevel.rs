use groggy::algorithms::community::Louvain;
use groggy::algorithms::{
    ensure_algorithms_registered, global_registry, Algorithm, Context, PipelineBuilder,
};
use groggy::subgraphs::Subgraph;
use groggy::{AttrValue, Graph};
use std::{cell::RefCell, rc::Rc};

fn ring(cliques: usize) -> Rc<RefCell<Graph>> {
    let mut graph = Graph::new_manual();
    graph.add_nodes(cliques * 3 + 1).unwrap();
    for c in 0..cliques {
        let a = c * 3;
        graph.add_edge(a, a + 1).unwrap();
        graph.add_edge(a + 1, a + 2).unwrap();
        graph.add_edge(a + 2, a).unwrap();
        graph.add_edge(a + 2, (a + 3) % (cliques * 3)).unwrap();
    }
    Rc::new(RefCell::new(graph))
}
fn run(g: &Rc<RefCell<Graph>>, phases: usize) -> (Vec<usize>, Context) {
    let sub = Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "test".into()).unwrap();
    let mut ctx = Context::new();
    Louvain::new(100, phases, 1., "c".into())
        .unwrap()
        .execute(&mut ctx, sub)
        .unwrap();
    let mut ids = g.borrow().node_ids();
    ids.sort_unstable();
    let labels = ids
        .iter()
        .map(|&node| {
            match g
                .borrow()
                .get_node_attr(node, &"c".into())
                .unwrap()
                .unwrap()
            {
                AttrValue::Int(v) => v as usize,
                AttrValue::SmallInt(v) => v as usize,
                other => panic!("{other:?}"),
            }
        })
        .collect();
    (labels, ctx)
}
// Original-graph objective from an independent adjacency-matrix calculation.
fn objective(g: &Rc<RefCell<Graph>>, labels: &[usize]) -> f64 {
    let g = g.borrow();
    let n = labels.len();
    let mut a = vec![vec![0.; n]; n];
    let mut degrees = vec![0.; n];
    for e in g.edge_ids() {
        let (u, v) = g.edge_endpoints(e).unwrap();
        a[u][v] += 1.;
        a[v][u] += 1.;
        degrees[u] += 1.;
        degrees[v] += 1.;
    }
    let twice_m = (2 * g.edge_ids().len()) as f64;
    let mut sum = 0.;
    for i in 0..n {
        for j in 0..n {
            if labels[i] == labels[j] {
                sum += a[i][j] - degrees[i] * degrees[j] / twice_m;
            }
        }
    }
    sum / twice_m
}
#[test]
fn coarse_levels_improve_over_local_moves_and_project_back_to_original_ids() {
    let g = ring(16);
    let (single, _) = run(&g, 1);
    let (multi, ctx) = run(&g, 10);
    assert!(objective(&g, &multi) > objective(&g, &single) + 1e-6);
    assert!(ctx.stat_snapshot()["louvain.total_phases"] > 1.);
    assert!(multi[..48].iter().all(|&c| c != multi[48]));
    assert_eq!(g.borrow().edge_ids().len(), 64);
    for c in 0..16 {
        assert_eq!(multi[c * 3], multi[c * 3 + 1]);
        assert_eq!(multi[c * 3 + 1], multi[c * 3 + 2]);
    }
    let mut previous = f64::NEG_INFINITY;
    for phase in 0..ctx.stat_snapshot()["louvain.total_phases"] as usize {
        let q = ctx.stat_snapshot()[&format!("louvain.phase_{phase}.modularity")];
        assert!(q >= previous - 1e-12);
        previous = q;
    }
    assert!((previous - objective(&g, &multi)).abs() < 1e-12);
    assert_eq!(run(&g, 10).0, multi);
}
#[test]
fn explicit_one_level_stays_available_and_registry_default_runs_multiple_levels() {
    ensure_algorithms_registered();
    let g = ring(16);
    let (single, _) = run(&g, 1);
    let sub = Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "test".into()).unwrap();
    let pipe = PipelineBuilder::new()
        .with_algorithm("community.louvain", |p| {
            p.set_text("output_attr", "c");
            p.set_int("max_iter", 100);
        })
        .build(global_registry())
        .unwrap();
    pipe.run(&mut Context::new(), sub).unwrap();
    let mut ids = g.borrow().node_ids();
    ids.sort_unstable();
    let labels: Vec<_> = ids
        .iter()
        .map(
            |&n| match g.borrow().get_node_attr(n, &"c".into()).unwrap().unwrap() {
                AttrValue::Int(v) => v as usize,
                AttrValue::SmallInt(v) => v as usize,
                _ => panic!(),
            },
        )
        .collect();
    assert!(objective(&g, &labels) > objective(&g, &single));
}

#[test]
fn original_id_gaps_loops_parallel_edges_and_isolates_survive_projection() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(7).unwrap();
    graph.remove_node(2).unwrap();
    graph.remove_node(6).unwrap();
    for (u, v) in [(0, 1), (0, 1), (3, 4), (4, 4)] {
        graph.add_edge(u, v).unwrap();
    }
    let g = Rc::new(RefCell::new(graph));
    let (labels, _) = run(&g, 10);
    assert_eq!(labels.len(), 5);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
    assert_ne!(labels[0], labels[2]);
    assert_ne!(labels[4], labels[0]);
    assert_ne!(labels[4], labels[2]);
    assert!(g.borrow().get_node_attr(2, &"c".into()).is_err());
}
