use groggy::{
    algorithms::centrality::{ClosenessCentrality, ClosenessDirection},
    AttrValue, Graph,
};
use std::sync::atomic::AtomicBool;
#[test]
fn captured_closeness_preserves_scope_properties_and_cancellation() {
    let mut graph = Graph::new_directed();
    let n = graph.add_nodes(3);
    let a = graph.add_edge(n[0], n[1]).unwrap();
    let b = graph.add_edge(n[1], n[2]).unwrap();
    graph.add_edge(n[0], n[2]).unwrap();
    graph
        .set_edge_attr(a, "w".into(), AttrValue::Int(2))
        .unwrap();
    let view = graph.read_view().unwrap().select_edges(&[a, b]).unwrap();
    graph
        .set_edge_attr(a, "w".into(), AttrValue::Int(0))
        .unwrap();
    drop(graph);
    let result = view
        .closeness(true, Some("w".into()), ClosenessDirection::Outgoing, false)
        .unwrap();
    assert!((result[0].1 - 5. / 6.).abs() < 1e-14);
    assert_eq!(result[2].1, 0.);
    let result = view
        .closeness(true, Some("w".into()), ClosenessDirection::Incoming, false)
        .unwrap();
    assert!((result[2].1 - 4. / 3.).abs() < 1e-14);
    assert_eq!(result[0].1, 0.);
    let algorithm = ClosenessCentrality::new(true, None, "unused".into());
    assert!(view
        .algorithm_input()
        .unwrap()
        .closeness(&algorithm, Some(&AtomicBool::new(true)))
        .is_err());
    let empty = Graph::new().read_view().unwrap().algorithm_input().unwrap();
    assert!(empty.closeness(&algorithm, None).unwrap().is_empty());
    assert!(empty
        .closeness(&algorithm, Some(&AtomicBool::new(true)))
        .is_err());
}
