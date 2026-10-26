use groggy::algorithms::community::LabelPropagation;
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

// Independent direct adjacency/value implementation; no CSR or interned IDs.
#[test]
fn all_three_node_directed_graphs_match_direct_value_propagation() {
    for mask in 0..512 {
        for initial in [[0, 1, 2], [0, 1, 0]] {
            let mut graph = Graph::new_directed_manual();
            graph.add_nodes(3).unwrap();
            let mut adjacency = vec![Vec::new(); 3];
            for (u, row) in adjacency.iter_mut().enumerate() {
                for v in 0..3 {
                    if mask & (1 << (3 * u + v)) != 0 {
                        graph.add_edge(u, v).unwrap();
                        row.push(v);
                    }
                }
                graph
                    .set_node_attr(u, "initial".into(), AttrValue::Int(initial[u]))
                    .unwrap();
            }
            let mut expected = initial;
            let mut dictionary = Vec::new();
            for label in initial {
                if !dictionary.contains(&label) {
                    dictionary.push(label);
                }
            }
            for _ in 0..10 {
                let mut changes = 0;
                for u in 0..3 {
                    let count = |label| {
                        adjacency[u]
                            .iter()
                            .filter(|&&v| expected[v] == label)
                            .count()
                    };
                    let max_count = dictionary.iter().map(|&label| count(label)).max().unwrap();
                    if max_count == 0 || count(expected[u]) == max_count {
                        continue;
                    }
                    expected[u] = *dictionary
                        .iter()
                        .find(|&&label| count(label) == max_count)
                        .unwrap();
                    changes += 1;
                }
                if changes == 0 {
                    break;
                }
            }
            let g = Rc::new(RefCell::new(graph));
            let sub =
                Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "test".into()).unwrap();
            LabelPropagation::new(10, 0., "label".into(), Some("initial".into()))
                .unwrap()
                .execute(&mut Context::new(), sub)
                .unwrap();
            for (node, want) in expected.iter().enumerate() {
                assert_eq!(
                    g.borrow().get_node_attr(node, &"label".into()).unwrap(),
                    Some(AttrValue::Int(*want)),
                    "mask={mask}, node={node}, initial={initial:?}"
                );
            }
        }
    }
}

#[test]
fn local_seed_is_repeatable_and_failure_preserves_labels() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(5).unwrap();
    graph.remove_node(1).unwrap();
    for (u, v) in [(0, 2), (0, 2), (2, 3), (3, 4), (4, 0), (4, 4)] {
        graph.add_edge(u, v).unwrap();
    }
    let g = Rc::new(RefCell::new(graph));
    let sub = Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "test".into()).unwrap();
    let algo = LabelPropagation::new(10, 0., "label".into(), None)
        .unwrap()
        .with_seed(Some(23));
    let labels = || [0, 2, 3, 4].map(|n| g.borrow().get_node_attr(n, &"label".into()).unwrap());
    fastrand::seed(5);
    let next = fastrand::u64(..);
    fastrand::seed(5);
    algo.execute(&mut Context::new(), sub.clone()).unwrap();
    assert_eq!(fastrand::u64(..), next);
    let before = labels();
    algo.execute(&mut Context::new(), sub.clone()).unwrap();
    assert_eq!(labels(), before);
    let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
    assert!(algo.execute(&mut ctx, sub.clone()).is_err());
    assert_eq!(labels(), before);
    g.borrow_mut().remove_node(0).unwrap();
    assert!(algo.execute(&mut Context::new(), sub).is_err());
}

#[test]
fn invalid_controls_and_empty_cancellation_reject() {
    for tolerance in [-1., 1.1, f64::NAN, f64::INFINITY] {
        assert!(LabelPropagation::new(10, tolerance, "c".into(), None).is_err());
    }
    ensure_algorithms_registered();
    for (key, value) in [("max_iter", 0), ("max_iter", -1), ("seed", -1)] {
        assert!(PipelineBuilder::new()
            .with_algorithm("community.lpa", |p| p.set_int(key, value))
            .build(global_registry())
            .is_err());
    }
    let g = Rc::new(RefCell::new(Graph::new_manual()));
    let sub = Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "empty".into()).unwrap();
    let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
    assert!(LabelPropagation::new(1, 0., "c".into(), None)
        .unwrap()
        .execute(&mut ctx, sub)
        .is_err());
}

#[test]
fn nonpersistent_empty_and_edgeless_outputs_retain_node_identity() {
    use groggy::algorithms::AlgorithmOutput;
    for n in [0, 1, 3] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(n).unwrap();
        let g = Rc::new(RefCell::new(graph));
        let sub =
            Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "test".into()).unwrap();
        let mut ctx = Context::new();
        ctx.set_persist_results(false);
        LabelPropagation::new(10, 0., "c".into(), None)
            .unwrap()
            .execute(&mut ctx, sub)
            .unwrap();
        let AlgorithmOutput::Components(groups) = ctx
            .take_outputs()
            .remove("community.lpa.components")
            .unwrap()
        else {
            panic!("expected component output")
        };
        assert_eq!(groups, (0..n).map(|v| vec![v]).collect::<Vec<_>>());
        for node in 0..n {
            assert!(g
                .borrow()
                .get_node_attr(node, &"c".into())
                .unwrap()
                .is_none());
        }
    }
}
