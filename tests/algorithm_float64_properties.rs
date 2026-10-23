use groggy::{
    algorithms::{
        centrality::{BetweennessCentrality, ClosenessCentrality, PageRank},
        pathfinding::DijkstraShortestPath,
        Algorithm, AlgorithmOutput, Context,
    },
    entities::{Edge, EntityEdge},
    subgraphs::Subgraph,
    traits::EdgeOperations,
    AttrValue, Graph,
};
use std::{cell::RefCell, rc::Rc};
fn selected(graph: &Rc<RefCell<Graph>>) -> Subgraph {
    Subgraph::from_nodes(graph.clone(), graph.borrow().node_ids_set(), "f64".into()).unwrap()
}
#[test]
fn persisted_scores_keep_the_exact_owned_f64_result() {
    let mut graph = Graph::new_directed();
    graph.add_nodes(4);
    graph
        .set_node_attr(0, "source".into(), AttrValue::Bool(true))
        .unwrap();
    for (u, v, weight) in [
        (0, 1, 1. + 2f64.powi(-40)),
        (1, 2, 2.),
        (0, 2, 4.),
        (2, 3, 1.),
    ] {
        let edge = graph.add_edge(u, v).unwrap();
        graph
            .set_edge_attr(edge, "w".into(), AttrValue::Float64(weight))
            .unwrap();
    }
    let graph = Rc::new(RefCell::new(graph));
    let algorithms: Vec<(Box<dyn Algorithm>, &str)> = vec![
        (
            Box::new(DijkstraShortestPath::new(
                "source".into(),
                Some("w".into()),
                "result".into(),
            )),
            "pathfinding.dijkstra.distances",
        ),
        (
            Box::new(ClosenessCentrality::new(
                false,
                Some("w".into()),
                "result".into(),
            )),
            "centrality.closeness.scores",
        ),
        (
            Box::new(BetweennessCentrality::new(true, None, "result".into())),
            "centrality.betweenness.scores",
        ),
        (
            Box::new(PageRank::new(0.85, 200, 1e-13, None, "result".into()).unwrap()),
            "centrality.pagerank.scores",
        ),
    ];
    for (algorithm, key) in algorithms {
        let mut context = Context::new();
        context.set_persist_results(false);
        algorithm.execute(&mut context, selected(&graph)).unwrap();
        let AlgorithmOutput::NodeScores(expected) = context.take_outputs().remove(key).unwrap()
        else {
            panic!("expected scores");
        };
        algorithm
            .execute(&mut Context::new(), selected(&graph))
            .unwrap();
        for (node, score) in expected {
            assert_eq!(
                graph
                    .borrow()
                    .get_node_attr(node, &"result".into())
                    .unwrap(),
                Some(AttrValue::Float64(score))
            );
        }
    }
}
#[test]
fn canonical_and_compatibility_edges_share_wide_weight_and_capacity_access() {
    let mut graph = Graph::new();
    graph.add_nodes(2);
    let id = graph.add_edge(0, 1).unwrap();
    let graph = Rc::new(RefCell::new(graph));
    let edge = Edge::new(id, graph.clone()).unwrap();
    let old = EntityEdge::new(id, graph.clone());
    for weight in [1. + 2f64.powi(-40), 1e300] {
        old.set_weight(weight).unwrap();
        assert_eq!(edge.weight().unwrap(), Some(weight));
        assert_eq!(old.weight().unwrap(), Some(weight));
        assert_eq!(
            graph.borrow().get_edge_attr(id, &"weight".into()).unwrap(),
            Some(AttrValue::Float64(weight))
        );
        edge.set_weight(-weight).unwrap();
        assert_eq!(old.weight().unwrap(), Some(-weight));
    }
    for value in [
        AttrValue::Float(0.5),
        AttrValue::Float64(1e300),
        AttrValue::Int(2),
        AttrValue::SmallInt(3),
    ] {
        graph
            .borrow_mut()
            .set_edge_attr(id, "capacity".into(), value)
            .unwrap();
        assert_eq!(edge.capacity().unwrap(), old.capacity().unwrap());
        assert!(edge.capacity().unwrap().is_some());
    }
    graph
        .borrow_mut()
        .set_edge_attr(id, "weight".into(), AttrValue::Text("unknown".into()))
        .unwrap();
    assert_eq!(edge.weight().unwrap(), None);
    assert_eq!(old.weight().unwrap(), None);
}
