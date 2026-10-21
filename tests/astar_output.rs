use groggy::algorithms::pathfinding::{AStarPathfinding, BfsTraversal};
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
fn output(ctx: &mut Context) -> Vec<(usize, i64)> {
    let AlgorithmOutput::NodeIntegers(rows) =
        ctx.take_outputs().remove("pathfinding.astar.path").unwrap()
    else {
        panic!("expected path indices")
    };
    rows
}
#[test]
fn every_small_graph_and_endpoint_pair_matches_floyd_warshall_cost() {
    for directed in [false, true] {
        for mask in 0..512 {
            let mut graph = if directed {
                Graph::new_directed()
            } else {
                Graph::new()
            };
            graph.add_nodes(3);
            let mut edge_cost = [[f64::INFINITY; 3]; 3];
            for bit in (0..9).rev() {
                if mask & (1 << bit) == 0 {
                    continue;
                }
                let (u, v) = (bit / 3, bit % 3);
                for weight in [9, bit % 3] {
                    let e = graph.add_edge(u, v).unwrap();
                    graph
                        .set_edge_attr(e, "w".into(), AttrValue::Int(weight as i64))
                        .unwrap();
                    edge_cost[u][v] = edge_cost[u][v].min(weight as f64);
                    if !directed {
                        edge_cost[v][u] = edge_cost[v][u].min(weight as f64);
                    }
                }
            }
            for node in 0..3 {
                graph
                    .set_node_attr(node, format!("s{node}"), AttrValue::Bool(false))
                    .unwrap();
                graph
                    .set_node_attr(node, format!("g{node}"), AttrValue::Null)
                    .unwrap();
            }
            let mut d = edge_cost;
            for (i, row) in d.iter_mut().enumerate() {
                row[i] = 0.;
            }
            for k in 0..3 {
                for u in 0..3 {
                    for v in 0..3 {
                        d[u][v] = d[u][v].min(d[u][k] + d[k][v]);
                    }
                }
            }
            let g = Rc::new(RefCell::new(graph));
            let mut warm = Context::new();
            warm.set_persist_results(false);
            let cached = BfsTraversal::new("s0".into(), "unused".into())
                .execute(&mut warm, selected(&g))
                .unwrap();
            for (source, row) in d.into_iter().enumerate() {
                for (goal, want) in row.into_iter().enumerate() {
                    let algo = AStarPathfinding::new(
                        format!("s{source}"),
                        format!("g{goal}"),
                        Some("w".into()),
                        None,
                        "out".into(),
                    );
                    let mut ctx = Context::new();
                    ctx.set_persist_results(false);
                    let result = algo.execute(&mut ctx, cached.clone());
                    assert!(ctx.call_counters().contains_key("astar.csr_cache_hit"));
                    if want.is_infinite() {
                        assert!(result.unwrap_err().to_string().contains("unreachable"));
                        assert!(ctx.take_outputs().is_empty());
                        continue;
                    }
                    result.unwrap();
                    let rows = output(&mut ctx);
                    assert_eq!(rows.first().unwrap().0, source);
                    assert_eq!(rows.last().unwrap().0, goal);
                    for (index, &(_, rank)) in rows.iter().enumerate() {
                        assert_eq!(rank, index as i64);
                    }
                    let actual: f64 = rows
                        .windows(2)
                        .map(|pair| edge_cost[pair[0].0][pair[1].0])
                        .sum();
                    assert_eq!(
                        actual, want,
                        "directed={directed},mask={mask},source={source},goal={goal}"
                    );
                    assert_eq!(ctx.stat_snapshot()["astar.path_cost"], want);
                }
            }
            for n in 0..3 {
                assert!(g
                    .borrow()
                    .get_node_attr(n, &"out".into())
                    .unwrap()
                    .is_none());
            }
        }
    }
}
#[test]
fn cancelled_trivial_paths_stale_and_open_selections_preserve_old_values() {
    let mut graph = Graph::new();
    graph.add_nodes(2);
    let e = graph.add_edge(0, 1).unwrap();
    graph
        .set_node_attr(0, "s".into(), AttrValue::Bool(true))
        .unwrap();
    graph
        .set_node_attr(0, "g".into(), AttrValue::Bool(true))
        .unwrap();
    graph
        .set_node_attr(0, "out".into(), AttrValue::Int(7))
        .unwrap();
    let g = Rc::new(RefCell::new(graph));
    let algo = AStarPathfinding::new("s".into(), "g".into(), None, None, "out".into());
    for persist in [false, true] {
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        ctx.set_persist_results(persist);
        assert!(algo
            .execute(&mut ctx, selected(&g))
            .unwrap_err()
            .to_string()
            .contains("cancelled"));
        assert!(ctx.take_outputs().is_empty());
    }
    let invalid = Subgraph::new(
        g.clone(),
        [0].into_iter().collect(),
        [e].into_iter().collect(),
        "bad".into(),
    );
    assert!(algo.execute(&mut Context::new(), invalid).is_err());
    let mut ctx = Context::new();
    ctx.set_persist_results(false);
    let cached = algo.execute(&mut ctx, selected(&g)).unwrap();
    assert_eq!(output(&mut ctx), vec![(0, 0)]);
    assert_eq!(ctx.stat_snapshot()["astar.path_cost"], 0.);
    g.borrow_mut().remove_edge(e).unwrap();
    assert!(algo.execute(&mut Context::new(), cached).is_err());
    assert_eq!(
        g.borrow().get_node_attr(0, &"out".into()).unwrap(),
        Some(AttrValue::Int(7))
    );
}
#[test]
fn sparse_path_rows_and_large_f64_costs_do_not_require_f32_distance_properties() {
    let mut graph = Graph::new_directed();
    graph.add_nodes(4);
    graph.remove_node(1).unwrap();
    graph
        .set_node_attr(3, "s".into(), AttrValue::Int(0))
        .unwrap();
    graph.set_node_attr(0, "g".into(), AttrValue::Null).unwrap();
    for (u, v) in [(3, 2), (2, 0)] {
        let e = graph.add_edge(u, v).unwrap();
        graph
            .set_edge_attr(e, "w".into(), AttrValue::Float(f32::MAX))
            .unwrap();
    }
    let g = Rc::new(RefCell::new(graph));
    let algo = AStarPathfinding::new("s".into(), "g".into(), Some("w".into()), None, "out".into());
    let mut ctx = Context::new();
    ctx.set_persist_results(false);
    let sub = algo.execute(&mut ctx, selected(&g)).unwrap();
    assert_eq!(output(&mut ctx), vec![(3, 0), (2, 1), (0, 2)]);
    assert_eq!(ctx.stat_snapshot()["astar.path_cost"], 2. * f32::MAX as f64);
    algo.execute(&mut Context::new(), sub).unwrap();
    for (node, want) in [(3, 0), (2, 1), (0, 2)] {
        assert_eq!(
            g.borrow().get_node_attr(node, &"out".into()).unwrap(),
            Some(AttrValue::Int(want))
        );
    }
}
