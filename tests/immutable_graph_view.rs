use groggy::{subgraphs::Subgraph, types::GraphType, AttrValue, Graph, GraphView};
use std::{cell::RefCell, collections::HashSet, rc::Rc};
#[test]
fn nested_views_share_frozen_state_and_preserve_parent_edge_scope() {
    let mut graph = Graph::new();
    let nodes = graph.add_nodes(3);
    let a = nodes[0];
    let b = nodes[1];
    let first = graph.add_edge(a, b).unwrap();
    let second = graph.add_edge(a, b).unwrap();
    let loop_edge = graph.add_edge(a, a).unwrap();
    graph
        .set_node_attr(a, "value".into(), AttrValue::Int(i64::MAX))
        .unwrap();
    graph
        .set_edge_attr(first, "weight".into(), AttrValue::Int(4))
        .unwrap();
    let all = graph.read_view().unwrap();
    let selected = all.select_edges(&[first, loop_edge, first]).unwrap();
    let child = selected.select_nodes(&[a, b, b]).unwrap();
    assert_eq!(child.edge_ids(), vec![first, loop_edge]);
    assert!(!child.contains_edge(second));
    assert_eq!(child.neighbors(a).unwrap(), vec![a, b]);
    assert_eq!(child.degree(a).unwrap(), 2);
    assert!(std::ptr::eq(
        all.read_state().snapshot(),
        child.read_state().snapshot()
    ));
    let copy = child.clone();
    assert!(std::ptr::eq(child.nodes(), copy.nodes()));
    assert!(std::ptr::eq(child.edges(), copy.edges()));
    graph
        .set_node_attr(a, "value".into(), AttrValue::Int(0))
        .unwrap();
    graph.remove_node(a).unwrap();
    assert_eq!(
        child.node_attr(a, &"value".into()).unwrap(),
        Some(AttrValue::Int(i64::MAX))
    );
    assert_eq!(
        child.edge_attr(first, &"weight".into()).unwrap(),
        Some(AttrValue::Int(4))
    );
    assert_eq!(
        child.select_nodes(&[a, b]).unwrap().neighbors(a).unwrap(),
        vec![a, b]
    );
    assert!(child.select_nodes(&[nodes[2]]).is_err());
    assert!(child.edge_endpoints(second).is_err());
}
#[test]
fn directed_degrees_count_selected_instances_and_loops_once() {
    let mut graph = Graph::new_with_type(GraphType::Directed);
    let nodes = graph.add_nodes(4);
    let a = nodes[0];
    let b = nodes[1];
    let c = nodes[2];
    let e1 = graph.add_edge(a, b).unwrap();
    let e2 = graph.add_edge(a, b).unwrap();
    let incoming = graph.add_edge(c, a).unwrap();
    let loop_edge = graph.add_edge(a, a).unwrap();
    let all = graph.read_view().unwrap();
    assert!(all.is_directed());
    assert_eq!(all.neighbors(a).unwrap(), vec![a, b, b]);
    assert_eq!(all.degree(a).unwrap(), 4);
    assert_eq!(all.in_degree(a).unwrap(), 2);
    assert_eq!(all.out_degree(a).unwrap(), 3);
    assert_eq!(
        all.incident_edges(a).unwrap(),
        vec![e1, e2, incoming, loop_edge]
    );
    assert_eq!(all.degree(nodes[3]).unwrap(), 0);
    let selected = all.select_nodes(&[a, b]).unwrap();
    assert_eq!(selected.in_degree(a).unwrap(), 1);
    assert_eq!(selected.degree(a).unwrap(), 3);
    assert!(selected.neighbors(b).unwrap().is_empty());
    assert_eq!(selected.in_degree(b).unwrap(), 2);
}
#[test]
fn smallest_graph_empty_views_and_membership_errors_are_explicit() {
    let empty = Graph::new().read_view().unwrap();
    assert_eq!(empty.node_count(), 0);
    assert_eq!(empty.edge_count(), 0);
    let mut graph = Graph::new();
    let node = graph.add_node();
    let view = graph.read_view().unwrap();
    assert_eq!(view.node_ids(), vec![node]);
    assert_eq!(view.edge_count(), 0);
    assert_eq!(view.degree(node).unwrap(), 0);
    assert_eq!(view.select_nodes(&[]).unwrap().node_count(), 0);
    assert!(view.node_attr(usize::MAX, &"value".into()).is_err());
    assert!(view.select_edges(&[usize::MAX]).is_err());
    assert!(GraphView::from_membership(
        view.read_state().clone(),
        HashSet::from([usize::MAX]),
        HashSet::new()
    )
    .is_err());
    let other = graph.add_node();
    let edge = graph.add_edge(node, other).unwrap();
    let state = graph.pin_current_state().unwrap();
    assert!(
        GraphView::from_membership(state, HashSet::from([node]), HashSet::from([edge])).is_err()
    );
}
#[test]
fn legacy_selection_can_be_explicitly_pinned_and_owned_projection_is_separate() {
    let graph = Rc::new(RefCell::new(Graph::new()));
    let nodes = graph.borrow_mut().add_nodes(2);
    let edge = graph.borrow_mut().add_edge(nodes[0], nodes[1]).unwrap();
    graph
        .borrow_mut()
        .set_node_attr(nodes[0], "value".into(), AttrValue::Int(1))
        .unwrap();
    let selection = Subgraph::new(
        graph.clone(),
        nodes.iter().copied().collect(),
        HashSet::from([edge]),
        "selected".into(),
    );
    let view = selection.pin_view().unwrap();
    let mut owned = Graph::from_snapshot(view.to_snapshot()).unwrap();
    owned
        .set_node_attr(nodes[0], "value".into(), AttrValue::Int(9))
        .unwrap();
    assert_eq!(
        view.node_attr(nodes[0], &"value".into()).unwrap(),
        Some(AttrValue::Int(1))
    );
    assert_eq!(
        graph
            .borrow()
            .get_node_attr(nodes[0], &"value".into())
            .unwrap(),
        Some(AttrValue::Int(1))
    );
    graph.borrow_mut().remove_edge(edge).unwrap();
    assert!(selection.pin_view().is_err());
    assert!(view.contains_edge(edge));
    drop(selection);
    drop(graph);
    assert_eq!(view.edge_endpoints(edge).unwrap(), (nodes[0], nodes[1]));
}
