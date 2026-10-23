use groggy::algorithms::pathfinding::DijkstraShortestPath;
use groggy::algorithms::{Algorithm, AlgorithmOutput, Context};
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
fn run(algo: &DijkstraShortestPath, sub: Subgraph) -> (Vec<(usize, f64)>, Context, Subgraph) {
    let mut ctx = Context::new();
    ctx.set_persist_results(false);
    let sub = algo.execute(&mut ctx, sub).unwrap();
    let AlgorithmOutput::NodeScores(rows) = ctx
        .take_outputs()
        .remove("pathfinding.dijkstra.distances")
        .unwrap()
    else {
        panic!("expected scores")
    };
    (rows, ctx, sub)
}
#[test]
fn exhaustive_zero_weight_multigraphs_match_floyd_warshall_for_every_source() {
    for directed in [false, true] {
        for mask in 0..512 {
            let mut graph = if directed {
                Graph::new_directed()
            } else {
                Graph::new()
            };
            graph.add_nodes(3);
            let mut d = [[f64::INFINITY; 3]; 3];
            for (i, row) in d.iter_mut().enumerate() {
                row[i] = 0.;
            }
            for bit in (0..9).rev() {
                if mask & (1 << bit) == 0 {
                    continue;
                }
                let (u, v) = (bit / 3, bit % 3);
                let expensive = graph.add_edge(u, v).unwrap();
                graph
                    .set_edge_attr(expensive, "w".into(), AttrValue::Int(9))
                    .unwrap();
                let e = graph.add_edge(u, v).unwrap();
                let weight = if bit % 3 == 0 { 1. } else { (bit % 4) as f64 };
                if bit % 3 != 0 {
                    graph
                        .set_edge_attr(e, "w".into(), AttrValue::Int(weight as i64))
                        .unwrap();
                }
                d[u][v] = d[u][v].min(weight);
                if !directed {
                    d[v][u] = d[v][u].min(weight);
                }
            }
            for k in 0..3 {
                for u in 0..3 {
                    for v in 0..3 {
                        d[u][v] = d[u][v].min(d[u][k] + d[k][v]);
                    }
                }
            }
            for source in 0..3 {
                graph
                    .set_node_attr(source, format!("s{source}"), AttrValue::Bool(false))
                    .unwrap();
            }
            let g = Rc::new(RefCell::new(graph));
            let mut sub = selected(&g);
            for (source, row) in d.into_iter().enumerate() {
                let expected: Vec<_> = row
                    .into_iter()
                    .enumerate()
                    .filter(|(_, value)| value.is_finite())
                    .collect();
                let algo =
                    DijkstraShortestPath::new(format!("s{source}"), Some("w".into()), "out".into());
                for repeat in 0..2 {
                    let (rows, ctx, next) = run(&algo, sub);
                    sub = next;
                    assert_eq!(
                        rows, expected,
                        "directed={directed}, mask={mask}, source={source}"
                    );
                    if repeat == 1 {
                        assert!(ctx.call_counters().contains_key("dijkstra.csr_cache_hit"));
                    }
                }
            }
            for node in 0..3 {
                assert!(g
                    .borrow()
                    .get_node_attr(node, &"out".into())
                    .unwrap()
                    .is_none());
            }
        }
    }
}
#[test]
fn missing_ambiguous_and_cancelled_sources_reject_without_result_writes() {
    for count in [0, 2] {
        let mut graph = Graph::new();
        graph.add_nodes(2);
        graph.add_edge(0, 1).unwrap();
        for node in 0..count {
            graph
                .set_node_attr(node, "source".into(), AttrValue::Null)
                .unwrap();
        }
        graph
            .set_node_attr(0, "out".into(), AttrValue::Int(7))
            .unwrap();
        let g = Rc::new(RefCell::new(graph));
        let algo = DijkstraShortestPath::new("source".into(), None, "out".into());
        let mut ctx = Context::new();
        ctx.set_persist_results(false);
        let error = algo.execute(&mut ctx, selected(&g)).unwrap_err();
        assert!(error.to_string().contains("exactly one"));
        assert!(ctx.take_outputs().is_empty());
        assert_eq!(
            g.borrow().get_node_attr(0, &"out".into()).unwrap(),
            Some(AttrValue::Int(7))
        );
    }
    let mut graph = Graph::new();
    graph.add_node();
    graph
        .set_node_attr(0, "source".into(), AttrValue::Int(0))
        .unwrap();
    let g = Rc::new(RefCell::new(graph));
    let algo = DijkstraShortestPath::new("source".into(), None, "out".into());
    let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
    assert!(algo.execute(&mut ctx, selected(&g)).is_err());
    assert!(g
        .borrow()
        .get_node_attr(0, &"out".into())
        .unwrap()
        .is_none());
    assert_eq!(run(&algo, selected(&g)).0, vec![(0, 0.)]);
    let empty = Rc::new(RefCell::new(Graph::new()));
    assert!(algo.execute(&mut Context::new(), selected(&empty)).is_err());
}
#[test]
fn invalid_unreachable_weights_open_and_stale_cached_selections_reject() {
    for bad in [
        AttrValue::Float(-1.),
        AttrValue::Float(f32::NAN),
        AttrValue::Float(f32::INFINITY),
        AttrValue::Bool(true),
        AttrValue::Null,
        AttrValue::Text("bad".into()),
    ] {
        let mut graph = Graph::new_directed();
        graph.add_nodes(3);
        graph
            .set_node_attr(0, "source".into(), AttrValue::Bool(true))
            .unwrap();
        let e = graph.add_edge(1, 2).unwrap();
        graph.set_edge_attr(e, "w".into(), bad).unwrap();
        let g = Rc::new(RefCell::new(graph));
        let mut ctx = Context::new();
        ctx.set_persist_results(false);
        assert!(
            DijkstraShortestPath::new("source".into(), Some("w".into()), "out".into())
                .execute(&mut ctx, selected(&g))
                .is_err()
        );
        assert!(ctx.take_outputs().is_empty());
    }
    let mut graph = Graph::new();
    graph.add_nodes(2);
    let e = graph.add_edge(0, 1).unwrap();
    graph
        .set_node_attr(0, "source".into(), AttrValue::Bool(true))
        .unwrap();
    let g = Rc::new(RefCell::new(graph));
    let algo = DijkstraShortestPath::new("source".into(), None, "out".into());
    let invalid = Subgraph::new(
        g.clone(),
        [0].into_iter().collect(),
        [e].into_iter().collect(),
        "open".into(),
    );
    assert!(algo.execute(&mut Context::new(), invalid).is_err());
    let (_, _, cached) = run(&algo, selected(&g));
    g.borrow_mut().remove_edge(e).unwrap();
    assert!(algo.execute(&mut Context::new(), cached).is_err());
    assert!(g
        .borrow()
        .get_node_attr(0, &"out".into())
        .unwrap()
        .is_none());
}
#[test]
fn f64_rows_preserve_large_distances_and_tiny_improvements_with_sparse_ids() {
    let mut graph = Graph::new_directed();
    graph.add_nodes(4);
    graph.remove_node(1).unwrap();
    graph
        .set_node_attr(0, "source".into(), AttrValue::Bool(true))
        .unwrap();
    for (u, v) in [(0, 2), (2, 3)] {
        let e = graph.add_edge(u, v).unwrap();
        graph
            .set_edge_attr(e, "w".into(), AttrValue::Float(f32::MAX))
            .unwrap();
    }
    let g = Rc::new(RefCell::new(graph));
    let algo = DijkstraShortestPath::new("source".into(), Some("w".into()), "out".into());
    assert_eq!(
        run(&algo, selected(&g)).0,
        vec![(0, 0.), (2, f32::MAX as f64), (3, 2. * f32::MAX as f64)]
    );
    algo.execute(&mut Context::new(), selected(&g)).unwrap();
    for (n, distance) in [(0, 0.), (2, f32::MAX as f64), (3, 2. * f32::MAX as f64)] {
        assert_eq!(g.borrow().get_node_attr(n, &"out".into()).unwrap(), Some(AttrValue::Float64(distance)));
    }
    let mut graph = Graph::new_directed();
    graph.add_nodes(3);
    graph
        .set_node_attr(0, "source".into(), AttrValue::Null)
        .unwrap();
    for (u, v, w) in [(0, 1, 5e-10_f32), (1, 2, 5e-10_f32), (0, 2, 2e-9_f32)] {
        let e = graph.add_edge(u, v).unwrap();
        graph
            .set_edge_attr(e, "w".into(), AttrValue::Float(w))
            .unwrap();
    }
    let g = Rc::new(RefCell::new(graph));
    let rows = run(&algo, selected(&g)).0;
    assert_eq!(rows[2].1, 2. * 5e-10_f32 as f64);
}
