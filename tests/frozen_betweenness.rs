use groggy::{algorithms::centrality::BetweennessCentrality, AttrValue, Graph};
use std::sync::atomic::AtomicBool;
#[test]
fn captured_brandes_keeps_scope_values_isolates_and_cancellation() {
    let mut graph = Graph::new_directed_manual();
    let n = graph.add_nodes(4).unwrap();
    let first = graph.add_edge(n[0], n[1]).unwrap();
    graph.add_edge(n[1], n[2]).unwrap();
    graph
        .set_edge_attr(first, "w".into(), AttrValue::Int(2))
        .unwrap();
    let view = graph.read_view().unwrap();
    graph.add_edge(n[0], n[2]).unwrap();
    graph
        .set_edge_attr(first, "w".into(), AttrValue::Int(0))
        .unwrap();
    drop(graph);
    assert_eq!(
        view.betweenness(false, Some("w".into())).unwrap(),
        vec![(n[0], 0.), (n[1], 1.), (n[2], 0.), (n[3], 0.)]
    );
    let algorithm = BetweennessCentrality::new(true, None, "unused".into());
    assert!(view
        .algorithm_input()
        .unwrap()
        .betweenness(&algorithm, Some(&AtomicBool::new(true)))
        .is_err());
    let empty = Graph::new_manual()
        .read_view()
        .unwrap()
        .algorithm_input()
        .unwrap();
    assert!(empty.betweenness(&algorithm, None).unwrap().is_empty());
    assert!(empty
        .betweenness(&algorithm, Some(&AtomicBool::new(true)))
        .is_err());
}
