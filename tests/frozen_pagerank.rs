use groggy::{algorithms::centrality::PageRank, AttrValue, Graph};
use std::sync::atomic::AtomicBool;

#[test]
fn captured_pagerank_uses_selected_occurrences_and_frozen_personalization() {
    let mut graph = Graph::new_directed_manual();
    let n = graph.add_nodes(3).unwrap();
    let first = graph.add_edge(n[0], n[1]).unwrap();
    let parallel = graph.add_edge(n[0], n[1]).unwrap();
    let other = graph.add_edge(n[0], n[2]).unwrap();
    graph.add_edge(n[1], n[0]).unwrap(); // Excluded.
    graph
        .set_node_attr(n[0], "p".into(), AttrValue::Int(1))
        .unwrap();
    graph
        .set_node_attr(n[1], "p".into(), AttrValue::Int(0))
        .unwrap();
    graph
        .set_node_attr(n[2], "p".into(), AttrValue::Int(0))
        .unwrap();
    let view = graph
        .read_view()
        .unwrap()
        .select_edges(&[first, parallel, other])
        .unwrap();
    graph
        .set_node_attr(n[0], "p".into(), AttrValue::Int(0))
        .unwrap();
    drop(graph);
    let result = view.pagerank(0.5, 1000, 1e-14, Some("p".into())).unwrap();
    assert!(result.converged);
    // Sink reset to node zero: x0=2/3, x1=2/9, x2=1/9.
    for ((_, actual), expected) in result.scores.iter().zip([2. / 3., 2. / 9., 1. / 9.]) {
        assert!((actual - expected).abs() < 1e-13);
    }
    let input = view.algorithm_input().unwrap();
    let algorithm = PageRank::new(0.85, 1, 1e-14, None, "unused".into()).unwrap();
    let limited = input.pagerank(&algorithm, None).unwrap();
    assert!(!limited.converged);
    assert_eq!(limited.iterations, 1);
    assert!(limited.max_update > 1e-14);
    assert!(input
        .pagerank(&algorithm, Some(&AtomicBool::new(true)))
        .is_err());
}

#[test]
fn empty_pagerank_is_explicit_and_cancellation_still_wins() {
    let graph = Graph::new_manual();
    let input = graph.read_view().unwrap().algorithm_input().unwrap();
    let algorithm = PageRank::new(0.85, 50, 1e-6, None, "unused".into()).unwrap();
    let result = input.pagerank(&algorithm, None).unwrap();
    assert!(result.scores.is_empty());
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.max_update, 0.);
    assert!(input
        .pagerank(&algorithm, Some(&AtomicBool::new(true)))
        .is_err());
}
