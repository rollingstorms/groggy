use groggy::algorithms::community::{ComponentMode, ConnectedComponents};
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
fn labels(g: &Shared, nodes: &[usize]) -> Vec<AttrValue> {
    nodes
        .iter()
        .map(|&n| g.borrow().get_node_attr(n, &"c".into()).unwrap().unwrap())
        .collect()
}

#[test]
fn exhaustive_three_node_graphs_match_independent_transitive_closure() {
    for directed in [false, true] {
        for mask in 0..512 {
            let mut graph = if directed {
                Graph::new_directed()
            } else {
                Graph::new()
            };
            graph.add_nodes(3);
            let mut adjacency = [[false; 3]; 3];
            for (u, row) in adjacency.iter_mut().enumerate() {
                row[u] = true;
            }
            for bit in 0..9 {
                let (u, v) = (bit / 3, bit % 3);
                if mask & (1 << bit) != 0 {
                    graph.add_edge(u, v).unwrap();
                    adjacency[u][v] = true;
                    if !directed {
                        adjacency[v][u] = true;
                    }
                }
            }
            let g = Rc::new(RefCell::new(graph));
            let sub = selected(&g);
            for mode in [
                ComponentMode::Undirected,
                ComponentMode::Weak,
                ComponentMode::Strong,
            ] {
                let mut reach = adjacency;
                if mode != ComponentMode::Strong {
                    for u in 0..3 {
                        for v in 0..3 {
                            reach[u][v] |= adjacency[v][u];
                        }
                    }
                }
                for k in 0..3 {
                    for u in 0..3 {
                        for v in 0..3 {
                            reach[u][v] |= reach[u][k] && reach[k][v];
                        }
                    }
                }
                let algo = ConnectedComponents::new(mode, "c".into());
                for _ in 0..2 {
                    algo.execute(&mut Context::new(), sub.clone()).unwrap();
                    let c = labels(&g, &[0, 1, 2]);
                    for u in 0..3 {
                        for v in 0..3 {
                            assert_eq!(
                                c[u] == c[v],
                                reach[u][v] && reach[v][u],
                                "directed={directed}, mask={mask}, mode={mode:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn cached_membership_rejects_stale_entities_and_cancellation() {
    for mode in [
        ComponentMode::Undirected,
        ComponentMode::Weak,
        ComponentMode::Strong,
    ] {
        let mut graph = Graph::new();
        graph.add_nodes(5);
        graph.remove_node(1).unwrap();
        let edge = graph.add_edge(2, 0).unwrap();
        graph.add_edge(2, 0).unwrap();
        graph.add_edge(2, 2).unwrap();
        let g = Rc::new(RefCell::new(graph));
        let sub = selected(&g);
        let algo = ConnectedComponents::new(mode, "c".into());
        algo.execute(&mut Context::new(), sub.clone()).unwrap();
        let before = labels(&g, &[0, 2, 3, 4]);
        assert_eq!(before[0], before[1]);
        assert_ne!(before[1], before[2]);
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        assert!(algo.execute(&mut ctx, sub.clone()).is_err());
        assert_eq!(labels(&g, &[0, 2, 3, 4]), before);
        g.borrow_mut().remove_edge(edge).unwrap();
        assert!(algo.execute(&mut Context::new(), sub).is_err());
        assert_eq!(labels(&g, &[0, 2, 3, 4]), before);
        // A new selection includes the remaining parallel instance.
        algo.execute(&mut Context::new(), selected(&g)).unwrap();
        assert_eq!(labels(&g, &[0, 2, 3, 4])[0], labels(&g, &[0, 2, 3, 4])[1]);
    }
}

#[test]
fn empty_singleton_closed_selection_and_nonpersistent_outputs() {
    for n in [0, 1, 3] {
        let mut graph = Graph::new_directed();
        graph.add_nodes(n);
        let g = Rc::new(RefCell::new(graph));
        for mode in [
            ComponentMode::Undirected,
            ComponentMode::Weak,
            ComponentMode::Strong,
        ] {
            let algo = ConnectedComponents::new(mode, "c".into());
            let mut ctx = Context::new();
            ctx.set_persist_results(false);
            algo.execute(&mut ctx, selected(&g)).unwrap();
            let AlgorithmOutput::Components(mut groups) = ctx
                .take_outputs()
                .remove("community.connected_components.components")
                .unwrap()
            else {
                panic!("expected component output")
            };
            groups.sort();
            assert_eq!(groups, (0..n).map(|v| vec![v]).collect::<Vec<_>>());
            assert!(g.borrow().node_ids().iter().all(|&id| g
                .borrow()
                .get_node_attr(id, &"c".into())
                .unwrap()
                .is_none()));
        }
    }
    let mut graph = Graph::new();
    graph.add_nodes(2);
    let edge = graph.add_edge(0, 1).unwrap();
    let g = Rc::new(RefCell::new(graph));
    let sub = Subgraph::new(
        g,
        [0].into_iter().collect(),
        [edge].into_iter().collect(),
        "invalid".into(),
    );
    assert!(ConnectedComponents::new(ComponentMode::Weak, "c".into())
        .execute(&mut Context::new(), sub)
        .is_err());
}

#[test]
fn long_directed_path_uses_an_explicit_stack() {
    let mut graph = Graph::new_directed();
    graph.add_nodes(10_000);
    for n in 1..10_000 {
        graph.add_edge(n, n - 1).unwrap();
    }
    let g = Rc::new(RefCell::new(graph));
    let mut ctx = Context::new();
    ctx.set_persist_results(false);
    ConnectedComponents::new(ComponentMode::Strong, "c".into())
        .execute(&mut ctx, selected(&g))
        .unwrap();
    let AlgorithmOutput::Components(groups) = ctx
        .take_outputs()
        .remove("community.connected_components.components")
        .unwrap()
    else {
        panic!("expected component output")
    };
    assert_eq!(groups.len(), 10_000);
}
