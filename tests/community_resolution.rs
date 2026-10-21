use groggy::algorithms::community::modularity::{
    modularity, modularity_delta, modularity_delta_with_resolution, modularity_with_resolution,
    ModularityData,
};
use groggy::algorithms::community::{Leiden, Louvain};
use groggy::algorithms::{
    ensure_algorithms_registered, global_registry, Algorithm, Context, PipelineBuilder,
};
use groggy::subgraphs::Subgraph;
use groggy::{AttrValue, Graph};
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
    sync::{atomic::AtomicBool, Arc},
};

// Independent adjacency-matrix definition, with a loop represented twice on the diagonal.
fn reference(labels: &[usize], edges: &[(usize, usize)], gamma: f64) -> f64 {
    let n = labels.len();
    if edges.is_empty() {
        return 0.;
    }
    let mut a = vec![vec![0.; n]; n];
    let mut degree = vec![0.; n];
    for &(u, v) in edges {
        a[u][v] += 1.;
        a[v][u] += 1.;
        degree[u] += 1.;
        degree[v] += 1.;
    }
    let twice_m = (2 * edges.len()) as f64;
    let mut q = 0.;
    for i in 0..n {
        for j in 0..n {
            if labels[i] == labels[j] {
                q += a[i][j] - gamma * degree[i] * degree[j] / twice_m;
            }
        }
    }
    q / twice_m
}
#[test]
fn resolution_move_gains_match_full_adjacency_definition_for_all_small_partitions() {
    let edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 3), (3, 3), (3, 1)];
    let data = ModularityData::new(&edges);
    let mut adjacency: HashMap<_, Vec<_>> = HashMap::new();
    for &(u, v) in &edges {
        adjacency.entry(u).or_default().push(v);
        adjacency.entry(v).or_default().push(u);
    }
    for code in 0_usize..81 {
        let mut code = code;
        let labels: Vec<_> = (0..4)
            .map(|_| {
                let l = code % 3;
                code /= 3;
                l
            })
            .collect();
        let partition: HashMap<_, _> = labels.iter().copied().enumerate().collect();
        let mut totals = HashMap::new();
        for (&n, &c) in &partition {
            *totals.entry(c).or_insert(0.) += data.degree(&n);
        }
        for gamma in [0.25, 1., 4.] {
            let before = reference(&labels, &edges, gamma);
            assert!(
                (modularity_with_resolution(&partition, &edges, &data, gamma) - before).abs()
                    < 1e-12
            );
            for node in 0..4 {
                for new in 0..4 {
                    let mut after = labels.clone();
                    after[node] = new;
                    let delta = modularity_delta_with_resolution(
                        node,
                        labels[node],
                        new,
                        &partition,
                        &adjacency,
                        &data,
                        &totals,
                        gamma,
                    );
                    assert!((delta - (reference(&after, &edges, gamma) - before)).abs() < 1e-12);
                    if gamma == 1. {
                        assert_eq!(
                            delta,
                            modularity_delta(
                                node,
                                labels[node],
                                new,
                                &partition,
                                &adjacency,
                                &data,
                                &totals,
                                &HashMap::new()
                            )
                        );
                        assert_eq!(
                            modularity(&partition, &edges, &data),
                            modularity_with_resolution(&partition, &edges, &data, 1.)
                        );
                    }
                }
            }
        }
    }
}
fn algorithm(leiden: bool, gamma: f64) -> Box<dyn Algorithm> {
    if leiden {
        Box::new(Leiden::new(100, 10, gamma, None, "c".into()).unwrap())
    } else {
        Box::new(Louvain::new(100, 1, gamma, "c".into()).unwrap())
    }
}
fn selected(g: &Rc<RefCell<Graph>>) -> Subgraph {
    Subgraph::from_nodes(g.clone(), g.borrow().node_ids_set(), "test".into()).unwrap()
}
fn read(g: &Rc<RefCell<Graph>>, nodes: &[usize]) -> Vec<AttrValue> {
    nodes
        .iter()
        .map(|&n| g.borrow().get_node_attr(n, &"c".into()).unwrap().unwrap())
        .collect()
}
#[test]
fn resolution_changes_local_moves_in_both_algorithms() {
    for leiden in [false, true] {
        for gamma in [0.5, 3.] {
            let mut graph = Graph::new();
            graph.add_nodes(2);
            graph.add_edge(1, 0).unwrap();
            let g = Rc::new(RefCell::new(graph));
            algorithm(leiden, gamma)
                .execute(&mut Context::new(), selected(&g))
                .unwrap();
            let c = read(&g, &[0, 1]);
            assert_eq!(c[0] == c[1], gamma == 0.5);
        }
    }
}
#[test]
fn loop_and_parallel_topology_matches_full_objective_and_keeps_isolates() {
    let edges = [
        (0, 0),
        (0, 0),
        (0, 1),
        (0, 1),
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 1),
        (3, 3),
    ];
    for leiden in [false, true] {
        let mut graph = Graph::new();
        graph.add_nodes(6);
        graph.remove_node(4).unwrap();
        for &(u, v) in &edges {
            graph.add_edge(u, v).unwrap();
        }
        let g = Rc::new(RefCell::new(graph));
        algorithm(leiden, 1.)
            .execute(&mut Context::new(), selected(&g))
            .unwrap();
        let values = read(&g, &[0, 1, 2, 3, 5]);
        assert!(values[..4].iter().all(|v| v != &values[4]));
        let mut map = Vec::new();
        let labels: Vec<_> = values
            .iter()
            .map(|v| {
                if let Some(i) = map.iter().position(|x| x == v) {
                    i
                } else {
                    map.push(v.clone());
                    map.len() - 1
                }
            })
            .collect();
        assert!(reference(&labels, &edges, 1.) >= reference(&[0, 1, 2, 3, 4], &edges, 1.) - 1e-12);
        for n in 0..4 {
            for c in 0..5 {
                let mut moved = labels.clone();
                moved[n] = c;
                assert!(reference(&moved, &edges, 1.) <= reference(&labels, &edges, 1.) + 1e-6);
            }
        }
        assert_eq!(g.borrow().edge_ids().len(), edges.len());
    }
}
#[test]
fn empty_edgeless_stale_directed_and_cancelled_inputs_have_explicit_behavior() {
    for leiden in [false, true] {
        for count in [0, 1, 3] {
            let mut graph = Graph::new();
            graph.add_nodes(count);
            let g = Rc::new(RefCell::new(graph));
            algorithm(leiden, 1.)
                .execute(&mut Context::new(), selected(&g))
                .unwrap();
            let values = read(&g, &(0..count).collect::<Vec<_>>());
            for (i, v) in values.iter().enumerate() {
                assert!(!values[..i].contains(v));
            }
        }
        let mut graph = Graph::new();
        graph.add_nodes(1);
        let g = Rc::new(RefCell::new(graph));
        let sub = selected(&g);
        g.borrow_mut().remove_node(0).unwrap();
        assert!(algorithm(leiden, 1.)
            .execute(&mut Context::new(), sub)
            .is_err());
        let g = Rc::new(RefCell::new(Graph::new_directed()));
        assert!(algorithm(leiden, 1.)
            .execute(&mut Context::new(), selected(&g))
            .is_err());
        let g = Rc::new(RefCell::new(Graph::new()));
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        assert!(algorithm(leiden, 1.)
            .execute(&mut ctx, selected(&g))
            .is_err());
    }
}
#[test]
fn constructors_and_registry_reject_invalid_parameters() {
    for gamma in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(Louvain::new(1, 1, gamma, "c".into()).is_err());
        assert!(Leiden::new(1, 1, gamma, None, "c".into()).is_err());
    }
    ensure_algorithms_registered();
    for id in ["community.louvain", "community.leiden"] {
        for param in ["max_iter", "max_phases"] {
            for value in [-1, 0] {
                assert!(PipelineBuilder::new()
                    .with_algorithm(id, |p| p.set_int(param, value))
                    .build(global_registry())
                    .is_err());
            }
        }
    }
}
#[test]
fn seeded_leiden_is_reproducible_without_global_rng_side_effects() {
    let mut graph = Graph::new();
    graph.add_nodes(8);
    for i in 0..8 {
        graph.add_edge(i, (i + 1) % 8).unwrap();
        graph.add_edge(i, (i + 3) % 8).unwrap();
    }
    let g = Rc::new(RefCell::new(graph));
    let algo = Leiden::new(50, 5, 1., Some(42), "c".into()).unwrap();
    let mut previous = None;
    for _ in 0..4 {
        fastrand::seed(123);
        let expected = fastrand::u64(..);
        fastrand::seed(123);
        algo.execute(&mut Context::new(), selected(&g)).unwrap();
        assert_eq!(fastrand::u64(..), expected);
        let labels = read(&g, &(0..8).collect::<Vec<_>>());
        if let Some(ref previous) = previous {
            assert_eq!(&labels, previous);
        }
        previous = Some(labels);
    }
}
