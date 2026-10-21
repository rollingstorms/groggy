use groggy::algorithms::pathfinding::DijkstraShortestPath;
use groggy::algorithms::steps::ShortestPathMapStep;
use groggy::algorithms::steps::{Step, StepScope, StepVariables};
use groggy::algorithms::{Algorithm, AlgorithmParamValue, Context};
use groggy::query::{PathFindingOptions, TraversalOptions};
use groggy::subgraphs::Subgraph;
use groggy::{AttrValue, Graph};
use std::{cell::RefCell, collections::HashSet, rc::Rc};
fn weighted() -> PathFindingOptions {
    PathFindingOptions {
        weight_attribute: Some("w".into()),
        ..Default::default()
    }
}
#[test]
fn direct_traversals_reject_unknown_and_removed_start_or_end_nodes() {
    let mut g = Graph::new();
    let n = g.add_node();
    g.remove_node(n).unwrap();
    for bad in [n, 999] {
        assert!(g.bfs(bad, TraversalOptions::default()).is_err());
        assert!(g.dfs(bad, TraversalOptions::default()).is_err());
        assert!(g
            .shortest_path(bad, bad, PathFindingOptions::default())
            .is_err());
        assert!(g.all_paths(bad, bad, 3).is_err());
    }
    let live = g.add_node();
    assert!(g.shortest_path(live, n, weighted()).is_err());
    assert!(g.all_paths(live, n, 3).is_err());
    assert_eq!(
        g.bfs(live, TraversalOptions::default()).unwrap().nodes,
        vec![live]
    );
    assert_eq!(
        g.shortest_path(live, live, weighted())
            .unwrap()
            .unwrap()
            .total_weight,
        0.
    );
}
#[test]
fn invalid_weights_reject_before_early_target_exit_and_ignoring_removed_edges() {
    let mut g = Graph::new_directed();
    let n = g.add_nodes(3);
    let edge = g.add_edge(n[1], n[2]).unwrap();
    for bad in [
        AttrValue::Float(-1.),
        AttrValue::Float(f32::NAN),
        AttrValue::Float(f32::INFINITY),
        AttrValue::Text("bad".into()),
        AttrValue::Null,
        AttrValue::Bool(true),
    ] {
        g.set_edge_attr(edge, "w".into(), bad).unwrap();
        assert!(g.shortest_path(n[0], n[0], weighted()).is_err());
    }
    g.remove_edge(edge).unwrap();
    assert!(g.shortest_path(n[0], n[0], weighted()).is_ok());
}
#[test]
fn compact_integer_missing_zero_and_parallel_weights_are_respected() {
    let mut g = Graph::new();
    let n = g.add_nodes(3);
    let expensive = g.add_edge(n[0], n[1]).unwrap();
    let cheap = g.add_edge(n[0], n[1]).unwrap();
    let end = g.add_edge(n[1], n[2]).unwrap();
    g.set_edge_attrs(std::collections::HashMap::from([(
        "w".into(),
        vec![
            (expensive, AttrValue::Int(9)),
            (cheap, AttrValue::Int(3)),
            (end, AttrValue::Float(0.)),
        ],
    )]))
    .unwrap();
    let path = g.shortest_path(n[2], n[0], weighted()).unwrap().unwrap();
    assert_eq!(path.edges, vec![end, cheap]);
    assert_eq!(path.total_weight, 3.);
    let unit = g.add_edge(n[0], n[2]).unwrap();
    let path = g.shortest_path(n[0], n[2], weighted()).unwrap().unwrap();
    assert_eq!(path.edges, vec![unit]);
    assert_eq!(path.total_weight, 1.);
}
fn algorithm_fixture(weight: AttrValue) -> (Rc<RefCell<Graph>>, Subgraph, usize) {
    let mut g = Graph::new();
    let n = g.add_nodes(2);
    g.set_node_attr(n[1], "source".into(), AttrValue::Bool(true))
        .unwrap();
    let e = g.add_edge(n[0], n[1]).unwrap();
    g.set_edge_attr(e, "w".into(), weight).unwrap();
    let g = Rc::new(RefCell::new(g));
    let sub = Subgraph::from_nodes(g.clone(), HashSet::from([n[0], n[1]]), "test".into()).unwrap();
    (g, sub, n[0])
}
#[test]
fn algorithm_and_pipeline_share_validation_and_symmetric_minimum_parallel_weights() {
    let (g, _sub, target) = algorithm_fixture(AttrValue::Int(8));
    let cheap = g.borrow_mut().add_edge(0, 1).unwrap();
    g.borrow_mut()
        .set_edge_attr(cheap, "w".into(), AttrValue::Int(2))
        .unwrap();
    let sub = Subgraph::from_nodes(g.clone(), HashSet::from([0, 1]), "test".into()).unwrap();
    let algorithm = DijkstraShortestPath::new("source".into(), Some("w".into()), "distance".into());
    algorithm.execute(&mut Context::new(), sub.clone()).unwrap();
    assert_eq!(
        g.borrow()
            .get_node_attr(target, &"distance".into())
            .unwrap(),
        Some(AttrValue::Float(2.))
    );
    let mut vars = StepVariables::default();
    vars.set_scalar("source", AlgorithmParamValue::Int(1));
    let step = ShortestPathMapStep::new("source".into(), Some("w".into()), "distances".into());
    step.apply(&mut Context::new(), &mut StepScope::new(&sub, &mut vars))
        .unwrap();
    assert_eq!(
        vars.node_map("distances").unwrap().get(&target),
        Some(&AlgorithmParamValue::Float(2.))
    );
    g.borrow_mut()
        .set_edge_attr(cheap, "w".into(), AttrValue::Float(-1.))
        .unwrap();
    assert!(algorithm.execute(&mut Context::new(), sub.clone()).is_err());
    assert!(step
        .apply(&mut Context::new(), &mut StepScope::new(&sub, &mut vars))
        .is_err());
    assert_eq!(
        g.borrow()
            .get_node_attr(target, &"distance".into())
            .unwrap(),
        Some(AttrValue::Float(2.))
    );
}
#[test]
fn tiny_distance_improvements_are_not_dropped_as_epsilon_noise() {
    let mut g = Graph::new_directed();
    let n = g.add_nodes(3);
    g.set_node_attr(n[0], "source".into(), AttrValue::Bool(true))
        .unwrap();
    for (u, v, w) in [(0, 1, 4e-20_f32), (0, 2, 1e-20_f32), (2, 1, 1e-20_f32)] {
        let e = g.add_edge(n[u], n[v]).unwrap();
        g.set_edge_attr(e, "w".into(), AttrValue::Float(w)).unwrap();
    }
    let sub = Subgraph::from_nodes(
        Rc::new(RefCell::new(g)),
        n.iter().copied().collect(),
        "test".into(),
    )
    .unwrap();
    let mut vars = StepVariables::default();
    vars.set_scalar("source", AlgorithmParamValue::Int(0));
    ShortestPathMapStep::new("source".into(), Some("w".into()), "d".into())
        .apply(&mut Context::new(), &mut StepScope::new(&sub, &mut vars))
        .unwrap();
    let AlgorithmParamValue::Float(distance) = vars.node_map("d").unwrap()[&n[1]] else {
        panic!("expected float")
    };
    assert!((distance - 2e-20).abs() < 1e-26, "{distance}");
}

#[test]
fn algorithm_distance_overflow_rejects_before_persisting_any_result() {
    let mut g = Graph::new_directed();
    let nodes = g.add_nodes(3);
    g.set_node_attr(nodes[0], "source".into(), AttrValue::Bool(true))
        .unwrap();
    for pair in nodes.windows(2) {
        let e = g.add_edge(pair[0], pair[1]).unwrap();
        g.set_edge_attr(e, "w".into(), AttrValue::Float(f32::MAX))
            .unwrap();
    }
    let g = Rc::new(RefCell::new(g));
    let sub =
        Subgraph::from_nodes(g.clone(), nodes.iter().copied().collect(), "test".into()).unwrap();
    assert!(
        DijkstraShortestPath::new("source".into(), Some("w".into()), "distance".into())
            .execute(&mut Context::new(), sub)
            .is_err()
    );
    for n in nodes {
        assert!(g
            .borrow()
            .get_node_attr(n, &"distance".into())
            .unwrap()
            .is_none());
    }
}

#[test]
fn direct_weighted_distances_match_independent_floyd_warshall_reference() {
    for directed in [false, true] {
        let mut g = if directed {
            Graph::new_directed()
        } else {
            Graph::new()
        };
        let nodes = g.add_nodes(7);
        let mut expected = vec![vec![f64::INFINITY; 7]; 7];
        for (i, row) in expected.iter_mut().enumerate() {
            row[i] = 0.;
        }
        for i in 0..25 {
            let from = (i * 3 + 1) % 7;
            let to = (i * 5 + 2) % 7;
            let weight = (i % 9) as i64;
            let edge = g.add_edge(nodes[from], nodes[to]).unwrap();
            g.set_edge_attr(edge, "w".into(), AttrValue::Int(weight))
                .unwrap();
            expected[from][to] = expected[from][to].min(weight as f64);
            if !directed {
                expected[to][from] = expected[to][from].min(weight as f64);
            }
        }
        for k in 0..7 {
            for i in 0..7 {
                for j in 0..7 {
                    expected[i][j] = expected[i][j].min(expected[i][k] + expected[k][j]);
                }
            }
        }
        for i in 0..7 {
            for j in 0..7 {
                let result = g.shortest_path(nodes[i], nodes[j], weighted()).unwrap();
                if expected[i][j].is_finite() {
                    assert_eq!(result.unwrap().total_weight, expected[i][j]);
                } else {
                    assert!(result.is_none());
                }
            }
        }
    }
}
