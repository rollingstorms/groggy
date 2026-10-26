use groggy::algorithms::pathfinding::AStarPathfinding;
use groggy::algorithms::{Algorithm, Context};
use groggy::subgraphs::Subgraph;
use groggy::{AttrValue, Graph};
use std::{cell::RefCell, rc::Rc};

fn run(g: Graph, heuristic: bool) -> anyhow::Result<Rc<RefCell<Graph>>> {
    let nodes = g.node_ids().into_iter().collect();
    let g = Rc::new(RefCell::new(g));
    let sub = Subgraph::from_nodes(g.clone(), nodes, "astar test".into())?;
    AStarPathfinding::new(
        "start".into(),
        "goal".into(),
        Some("w".into()),
        heuristic.then(|| "h".into()),
        "order".into(),
    )
    .execute(&mut Context::new(), sub)?;
    Ok(g)
}
fn fixture(directed: bool) -> Graph {
    let mut g = if directed {
        Graph::new_directed_manual()
    } else {
        Graph::new_manual()
    };
    g.add_nodes(3).unwrap();
    g.set_node_attr(2, "start".into(), AttrValue::Bool(true))
        .unwrap();
    g.set_node_attr(0, "goal".into(), AttrValue::Bool(true))
        .unwrap();
    for (u, v, w) in [(0, 2, 9), (0, 1, 2), (1, 2, 2), (0, 1, 8)] {
        let e = g.add_edge(u, v).unwrap();
        g.set_edge_attr(e, "w".into(), AttrValue::Int(w)).unwrap();
    }
    g
}
#[test]
fn reverse_parallel_compact_weights_choose_the_shortest_node_path() {
    let g = run(fixture(false), false).unwrap();
    for (node, order) in [(2, 0), (1, 1), (0, 2)] {
        assert_eq!(
            g.borrow().get_node_attr(node, &"order".into()).unwrap(),
            Some(AttrValue::Int(order))
        );
    }
    assert!(run(fixture(true), false).is_err());
}
#[test]
fn consistent_heuristic_accepts_compact_integer_values() {
    let mut g = fixture(false);
    for (n, h) in [(0, 0), (1, 2), (2, 4)] {
        g.set_node_attr(n, "h".into(), AttrValue::Int(h)).unwrap();
    }
    assert!(run(g, true).is_ok());
}
#[test]
fn invalid_heuristics_reject_before_result_writes() {
    for bad in [
        AttrValue::Float(f32::NAN),
        AttrValue::Float(f32::INFINITY),
        AttrValue::Int(-1),
        AttrValue::Null,
        AttrValue::Bool(true),
        AttrValue::Int(100),
    ] {
        let mut g = fixture(false);
        g.set_node_attr(1, "h".into(), bad).unwrap();
        assert!(run(g, true).is_err());
    }
    let mut g = fixture(false);
    g.set_node_attr(0, "h".into(), AttrValue::Int(1)).unwrap();
    assert!(run(g, true).is_err());
}
#[test]
fn tiny_improvements_are_retained() {
    let mut g = fixture(false);
    for (e, w) in [(0, 4e-20), (1, 1e-20), (2, 1e-20), (3, 8e-20)] {
        g.set_edge_attr(e, "w".into(), AttrValue::Float(w)).unwrap();
    }
    let g = run(g, false).unwrap();
    assert_eq!(
        g.borrow().get_node_attr(1, &"order".into()).unwrap(),
        Some(AttrValue::Int(1))
    );
}
#[test]
fn ambiguous_markers_and_bad_weights_reject_even_for_trivial_paths() {
    let mut g = fixture(false);
    g.set_node_attr(1, "start".into(), AttrValue::Bool(true))
        .unwrap();
    assert!(run(g, false).is_err());
    let mut g = Graph::new_manual();
    g.add_nodes(2).unwrap();
    for attr in ["start", "goal"] {
        g.set_node_attr(0, attr.into(), AttrValue::Bool(true))
            .unwrap();
    }
    let e = g.add_edge(0, 1).unwrap();
    g.set_edge_attr(e, "w".into(), AttrValue::Int(-1)).unwrap();
    assert!(run(g, false).is_err());
}
