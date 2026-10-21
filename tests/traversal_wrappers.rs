use groggy::algorithms::pathfinding::{BfsTraversal, DfsTraversal};
use groggy::algorithms::{Algorithm, AlgorithmOutput, Context};
use groggy::subgraphs::Subgraph;
use groggy::{AttrValue, Graph};
use std::{
    cell::RefCell,
    collections::VecDeque,
    rc::Rc,
    sync::{atomic::AtomicBool, Arc},
};
type Shared = Rc<RefCell<Graph>>;
fn selected(g: &Shared) -> Subgraph {
    Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "test".into()).unwrap()
}
fn run(algo: &dyn Algorithm, sub: Subgraph, key: &str) -> (Vec<(usize, i64)>, Context, Subgraph) {
    let mut ctx = Context::new();
    ctx.set_persist_results(false);
    let result = algo.execute(&mut ctx, sub).unwrap();
    let AlgorithmOutput::NodeIntegers(rows) = ctx.take_outputs().remove(key).unwrap() else {
        panic!("expected exact integer output")
    };
    (rows, ctx, result)
}
fn recursive_order(u: usize, adjacency: &[Vec<usize>], seen: &mut [bool], order: &mut Vec<usize>) {
    if seen[u] {
        return;
    }
    seen[u] = true;
    order.push(u);
    for &v in &adjacency[u] {
        recursive_order(v, adjacency, seen, order);
    }
}
#[test]
fn exhaustive_directed_and_undirected_graphs_match_independent_traversals() {
    for directed in [false, true] {
        for mask in 0..512 {
            let mut graph = if directed {
                Graph::new_directed()
            } else {
                Graph::new()
            };
            graph.add_nodes(3);
            let mut adjacency = vec![Vec::new(); 3];
            // Reverse insertion deliberately disagrees with ascending neighbor order.
            for bit in (0..9).rev() {
                if mask & (1 << bit) == 0 {
                    continue;
                }
                let (u, v) = (bit / 3, bit % 3);
                for _ in 0..=bit % 2 {
                    graph.add_edge(u, v).unwrap();
                }
                adjacency[u].push(v);
                if !directed {
                    adjacency[v].push(u);
                }
            }
            for row in &mut adjacency {
                row.sort_unstable();
                row.dedup();
            }
            for source in 0..3 {
                graph
                    .set_node_attr(source, format!("start_{source}"), AttrValue::Bool(false))
                    .unwrap();
            }
            let g = Rc::new(RefCell::new(graph));
            let mut sub = selected(&g);
            for source in 0..3 {
                let mut distances = [None; 3];
                let mut queue = VecDeque::from([source]);
                distances[source] = Some(0);
                while let Some(u) = queue.pop_front() {
                    for &v in &adjacency[u] {
                        if distances[v].is_none() {
                            distances[v] = Some(distances[u].unwrap() + 1);
                            queue.push_back(v);
                        }
                    }
                }
                let expected_bfs: Vec<_> = distances
                    .into_iter()
                    .enumerate()
                    .filter_map(|(node, d)| d.map(|d| (node, d)))
                    .collect();
                let mut order = Vec::new();
                recursive_order(source, &adjacency, &mut [false; 3], &mut order);
                let expected_dfs: Vec<_> = order
                    .into_iter()
                    .enumerate()
                    .map(|(rank, node)| (node, rank as i64))
                    .collect();
                let bfs = BfsTraversal::new(format!("start_{source}"), "bfs".into());
                let dfs = DfsTraversal::new(format!("start_{source}"), "dfs".into());
                for _ in 0..2 {
                    let (rows, _, next) = run(&bfs, sub, "pathfinding.bfs.distances");
                    assert_eq!(
                        rows, expected_bfs,
                        "BFS mask={mask}, directed={directed}, source={source}"
                    );
                    let (rows, ctx, next) = run(&dfs, next, "pathfinding.dfs.order");
                    sub = next;
                    assert_eq!(
                        rows, expected_dfs,
                        "DFS mask={mask}, directed={directed}, source={source}"
                    );
                    assert!(ctx.call_counters().contains_key("dfs.csr_cache_hit"));
                }
            }
            for node in 0..3 {
                assert!(g
                    .borrow()
                    .get_node_attr(node, &"bfs".into())
                    .unwrap()
                    .is_none());
                assert!(g
                    .borrow()
                    .get_node_attr(node, &"dfs".into())
                    .unwrap()
                    .is_none());
            }
        }
    }
}
#[test]
fn missing_or_ambiguous_markers_reject_before_writes_and_empty_has_no_source() {
    for marker_count in [0, 2] {
        let mut graph = Graph::new();
        graph.add_nodes(3);
        graph.add_edge(0, 1).unwrap();
        for node in 0..marker_count {
            graph
                .set_node_attr(node, "start".into(), AttrValue::Null)
                .unwrap();
        }
        graph
            .set_node_attr(0, "out".into(), AttrValue::Int(7))
            .unwrap();
        let g = Rc::new(RefCell::new(graph));
        for algo in [
            Box::new(BfsTraversal::new("start".into(), "out".into())) as Box<dyn Algorithm>,
            Box::new(DfsTraversal::new("start".into(), "out".into())),
        ] {
            let err = algo.execute(&mut Context::new(), selected(&g)).unwrap_err();
            assert!(err.to_string().contains("exactly one"));
            assert_eq!(
                g.borrow().get_node_attr(0, &"out".into()).unwrap(),
                Some(AttrValue::Int(7))
            );
        }
    }
    let g = Rc::new(RefCell::new(Graph::new()));
    assert!(BfsTraversal::new("start".into(), "out".into())
        .execute(&mut Context::new(), selected(&g))
        .is_err());
}
#[test]
fn stale_open_and_cancelled_selections_fail_without_outputs() {
    for dfs in [false, true] {
        let mut graph = Graph::new();
        graph.add_nodes(2);
        let e = graph.add_edge(0, 1).unwrap();
        graph
            .set_node_attr(0, "start".into(), AttrValue::Int(0))
            .unwrap();
        let g = Rc::new(RefCell::new(graph));
        let sub = selected(&g);
        let algo: Box<dyn Algorithm> = if dfs {
            Box::new(DfsTraversal::new("start".into(), "out".into()))
        } else {
            Box::new(BfsTraversal::new("start".into(), "out".into()))
        };
        let invalid = Subgraph::new(
            g.clone(),
            [0].into_iter().collect(),
            [e].into_iter().collect(),
            "open".into(),
        );
        assert!(algo.execute(&mut Context::new(), invalid).is_err());
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        ctx.set_persist_results(false);
        assert!(algo.execute(&mut ctx, sub.clone()).is_err());
        assert!(ctx.take_outputs().is_empty());
        run(
            algo.as_ref(),
            sub.clone(),
            if dfs {
                "pathfinding.dfs.order"
            } else {
                "pathfinding.bfs.distances"
            },
        );
        g.borrow_mut().remove_edge(e).unwrap();
        assert!(algo.execute(&mut Context::new(), sub).is_err());
        assert!(g
            .borrow()
            .get_node_attr(0, &"out".into())
            .unwrap()
            .is_none());
    }
}
#[test]
fn sparse_ids_singletons_and_long_paths_preserve_exact_integer_rows() {
    let mut graph = Graph::new_directed();
    graph.add_nodes(4);
    graph.remove_node(1).unwrap();
    graph.add_edge(3, 2).unwrap();
    graph.add_edge(2, 0).unwrap();
    graph
        .set_node_attr(3, "start".into(), AttrValue::Null)
        .unwrap();
    let g = Rc::new(RefCell::new(graph));
    let sub = selected(&g);
    assert_eq!(
        run(
            &BfsTraversal::new("start".into(), "out".into()),
            sub.clone(),
            "pathfinding.bfs.distances"
        )
        .0,
        vec![(0, 2), (2, 1), (3, 0)]
    );
    assert_eq!(
        run(
            &DfsTraversal::new("start".into(), "out".into()),
            sub,
            "pathfinding.dfs.order"
        )
        .0,
        vec![(3, 0), (2, 1), (0, 2)]
    );
    let mut graph = Graph::new_directed();
    graph.add_nodes(20_000);
    for node in 1..20_000 {
        graph.add_edge(node - 1, node).unwrap();
    }
    graph
        .set_node_attr(0, "start".into(), AttrValue::Bool(true))
        .unwrap();
    let g = Rc::new(RefCell::new(graph));
    let rows = run(
        &DfsTraversal::new("start".into(), "out".into()),
        selected(&g),
        "pathfinding.dfs.order",
    )
    .0;
    assert_eq!(rows.len(), 20_000);
    assert_eq!(rows[19_999], (19_999, 19_999));
    let mut graph = Graph::new();
    graph.add_node();
    graph
        .set_node_attr(0, "start".into(), AttrValue::Bool(false))
        .unwrap();
    let g = Rc::new(RefCell::new(graph));
    assert_eq!(
        run(
            &BfsTraversal::new("start".into(), "out".into()),
            selected(&g),
            "pathfinding.bfs.distances"
        )
        .0,
        vec![(0, 0)]
    );
}
