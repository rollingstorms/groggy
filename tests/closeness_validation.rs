use groggy::algorithms::centrality::{ClosenessCentrality, ClosenessDirection};
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
fn scores(
    g: &Shared,
    harmonic: bool,
    incoming: bool,
    normalized: bool,
    weighted: bool,
) -> Vec<(usize, f64)> {
    let mut ctx = Context::new();
    ctx.set_persist_results(false);
    let algorithm = ClosenessCentrality::new(harmonic, weighted.then(|| "w".into()), "c".into())
        .with_direction(if incoming {
            ClosenessDirection::Incoming
        } else {
            ClosenessDirection::Outgoing
        })
        .with_normalization(normalized);
    algorithm.execute(&mut ctx, selected(g)).unwrap();
    let AlgorithmOutput::NodeScores(values) = ctx
        .take_outputs()
        .remove("centrality.closeness.scores")
        .unwrap()
    else {
        panic!("expected scores")
    };
    let captured = g
        .borrow()
        .read_view()
        .unwrap()
        .algorithm_input()
        .unwrap()
        .closeness(&algorithm, None)
        .unwrap();
    assert_eq!(captured.len(), values.len());
    for ((node, actual), (expected_node, expected)) in captured.iter().zip(&values) {
        assert_eq!(node, expected_node);
        assert!((actual - expected).abs() <= 1e-12 * expected.abs().max(1.0));
    }
    values
}
#[test]
fn exhaustive_small_graphs_match_floyd_warshall_for_all_modes() {
    for directed in [false, true] {
        for weighted in [false, true] {
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
                for bit in 0..9 {
                    if mask & (1 << bit) == 0 {
                        continue;
                    }
                    let (u, v) = (bit / 3, bit % 3);
                    let e = graph.add_edge(u, v).unwrap();
                    let length = if weighted { (bit % 4 + 1) as f64 } else { 1. };
                    graph
                        .set_edge_attr(e, "w".into(), AttrValue::Int(length as i64))
                        .unwrap();
                    d[u][v] = d[u][v].min(length);
                    if !directed {
                        d[v][u] = d[v][u].min(length);
                    }
                }
                for k in 0..3 {
                    for u in 0..3 {
                        for v in 0..3 {
                            d[u][v] = d[u][v].min(d[u][k] + d[k][v]);
                        }
                    }
                }
                let g = Rc::new(RefCell::new(graph));
                for harmonic in [false, true] {
                    for incoming in [false, true] {
                        for normalized in [false, true] {
                            for (u, got) in scores(&g, harmonic, incoming, normalized, weighted) {
                                let distances: Vec<_> = (0..3)
                                    .filter(|&v| v != u)
                                    .map(|v| if incoming { d[v][u] } else { d[u][v] })
                                    .filter(|v| v.is_finite())
                                    .collect();
                                let sum: f64 = distances
                                    .iter()
                                    .map(|&x| if harmonic { 1. / x } else { x })
                                    .sum();
                                let raw = if harmonic {
                                    sum
                                } else if distances.is_empty() {
                                    0.
                                } else {
                                    distances.len() as f64 / sum
                                };
                                let want = if normalized {
                                    raw * if harmonic {
                                        0.5
                                    } else {
                                        distances.len() as f64 / 2.
                                    }
                                } else {
                                    raw
                                };
                                assert!((got-want).abs()<1e-12,"mask={mask}, directed={directed}, weighted={weighted}, harmonic={harmonic}, incoming={incoming}, normalized={normalized}, u={u}");
                            }
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn invalid_lengths_and_persistent_overflow_preserve_previous_values() {
    for bad in [
        AttrValue::Int(0),
        AttrValue::Int(-1),
        AttrValue::Float(f32::NAN),
        AttrValue::Float(f32::INFINITY),
        AttrValue::Bool(true),
        AttrValue::Null,
        AttrValue::Text("bad".into()),
    ] {
        let mut graph = Graph::new();
        graph.add_nodes(2);
        let e = graph.add_edge(0, 1).unwrap();
        graph.set_edge_attr(e, "w".into(), bad).unwrap();
        graph
            .set_node_attr(0, "c".into(), AttrValue::Int(7))
            .unwrap();
        let g = Rc::new(RefCell::new(graph));
        for harmonic in [false, true] {
            assert!(
                ClosenessCentrality::new(harmonic, Some("w".into()), "c".into())
                    .execute(&mut Context::new(), selected(&g))
                    .is_err()
            );
        }
        assert_eq!(
            g.borrow().get_node_attr(0, &"c".into()).unwrap(),
            Some(AttrValue::Int(7))
        );
    }
    let mut graph = Graph::new();
    graph.add_nodes(2);
    let e = graph.add_edge(0, 1).unwrap();
    graph
        .set_edge_attr(e, "w".into(), AttrValue::Float(f32::from_bits(1)))
        .unwrap();
    let g = Rc::new(RefCell::new(graph));
    for harmonic in [false, true] {
        let value = scores(&g, harmonic, false, false, true)[0].1;
        assert!(value.is_finite() && value > f32::MAX as f64);
        assert!(
            ClosenessCentrality::new(harmonic, Some("w".into()), "c".into())
                .execute(&mut Context::new(), selected(&g))
                .is_err()
        );
        assert!(g.borrow().get_node_attr(0, &"c".into()).unwrap().is_none());
    }
}
#[test]
fn empty_singleton_sparse_parallel_and_cached_direction_cases() {
    for n in [0, 1, 3] {
        let mut graph = Graph::new();
        graph.add_nodes(n);
        let g = Rc::new(RefCell::new(graph));
        assert_eq!(
            scores(&g, true, false, true, false),
            (0..n).map(|n| (n, 0.)).collect::<Vec<_>>()
        );
    }
    let mut graph = Graph::new_directed();
    graph.add_nodes(4);
    graph.remove_node(1).unwrap();
    let expensive = graph.add_edge(0, 2).unwrap();
    graph
        .set_edge_attr(expensive, "w".into(), AttrValue::Int(8))
        .unwrap();
    let cheap = graph.add_edge(0, 2).unwrap();
    graph
        .set_edge_attr(cheap, "w".into(), AttrValue::Int(2))
        .unwrap();
    graph.add_edge(2, 3).unwrap();
    graph.add_edge(2, 2).unwrap();
    let g = Rc::new(RefCell::new(graph));
    let mut sub = selected(&g);
    for incoming in [true, false, true, false] {
        let mut ctx = Context::new();
        ctx.set_persist_results(false);
        sub = ClosenessCentrality::new(true, Some("w".into()), "c".into())
            .with_direction(if incoming {
                ClosenessDirection::Incoming
            } else {
                ClosenessDirection::Outgoing
            })
            .execute(&mut ctx, sub)
            .unwrap();
        let AlgorithmOutput::NodeScores(values) = ctx
            .take_outputs()
            .remove("centrality.closeness.scores")
            .unwrap()
        else {
            panic!("expected scores")
        };
        assert_eq!(
            values.iter().map(|v| v.0).collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
        let want = if incoming {
            [0., 0.5, 4. / 3.]
        } else {
            [5. / 6., 1., 0.]
        };
        for ((_, v), w) in values.iter().zip(want) {
            assert!((v - w).abs() < 1e-12);
        }
    }
}
#[test]
fn stale_open_and_cancelled_selections_reject_before_writes() {
    let mut graph = Graph::new();
    graph.add_nodes(2);
    let e = graph.add_edge(0, 1).unwrap();
    let g = Rc::new(RefCell::new(graph));
    let sub = selected(&g);
    let algo = ClosenessCentrality::new(true, None, "c".into());
    let invalid = Subgraph::new(
        g.clone(),
        [0].into_iter().collect(),
        [e].into_iter().collect(),
        "bad".into(),
    );
    assert!(algo.execute(&mut Context::new(), invalid).is_err());
    let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
    assert!(algo.execute(&mut ctx, sub.clone()).is_err());
    g.borrow_mut().remove_edge(e).unwrap();
    assert!(algo.execute(&mut Context::new(), sub).is_err());
    assert!(g.borrow().get_node_attr(0, &"c".into()).unwrap().is_none());
}
