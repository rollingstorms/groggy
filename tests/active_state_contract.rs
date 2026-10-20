use groggy::storage::edge::EntityEdge;
use groggy::storage::node::EntityNode;
use groggy::traits::{EdgeOperations, GraphEntity, NodeOperations};
use groggy::types::AttrValue;
use groggy::Graph;
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn live_reads_exclude_removed_edges_and_nodes() {
    let mut graph = Graph::new();
    let source = graph.add_node();
    let target = graph.add_node();
    let edge = graph.add_edge(source, target).unwrap();

    assert_eq!(graph.incident_edges(source).unwrap(), vec![edge]);
    graph.remove_edge(edge).unwrap();
    assert!(!graph.contains_edge(edge));
    assert!(graph.edge_ids().is_empty());
    assert!(graph.incident_edges(source).unwrap().is_empty());
    assert!(graph.edge_endpoints(edge).is_err());
    assert_eq!(graph.get_edges_attrs(&"weight".into(), &[edge]).unwrap(), vec![None]);

    graph.remove_node(source).unwrap();
    assert!(!graph.contains_node(source));
    assert_eq!(graph.node_ids(), vec![target]);
    assert!(graph.incident_edges(source).is_err());
    assert!(graph.neighbors(source).is_err());
    assert!(graph.get_node_attr(source, &"entity_type".into()).is_err());
    assert_eq!(
        graph.get_nodes_attrs(&"entity_type".into(), &[source]).unwrap(),
        vec![None]
    );
}

#[test]
fn entity_handles_cannot_read_inactive_values_from_the_pool() {
    let graph = Rc::new(RefCell::new(Graph::new()));
    let source = graph.borrow_mut().add_node();
    let target = graph.borrow_mut().add_node();
    let edge = graph.borrow_mut().add_edge(source, target).unwrap();
    graph
        .borrow_mut()
        .set_node_attr(source, "name".into(), AttrValue::Text("Alice".into()))
        .unwrap();
    graph
        .borrow_mut()
        .set_edge_attr(edge, "weight".into(), AttrValue::Int(3))
        .unwrap();

    let node_handle = EntityNode::new(source, graph.clone());
    let edge_handle = EntityEdge::new(edge, graph.clone());
    assert_eq!(
        node_handle.get_attribute(&"name".into()).unwrap(),
        Some(AttrValue::Text("Alice".into()))
    );
    assert_eq!(
        edge_handle.get_attribute(&"weight".into()).unwrap(),
        Some(AttrValue::Int(3))
    );

    graph.borrow_mut().remove_edge(edge).unwrap();
    assert!(!edge_handle.is_active());
    assert!(edge_handle.get_attribute(&"weight".into()).is_err());
    assert!(edge_handle.edge_attributes().is_err());

    graph.borrow_mut().remove_node(source).unwrap();
    assert!(!node_handle.is_active());
    assert!(node_handle.get_attribute(&"name".into()).is_err());
    assert!(node_handle.node_attributes().is_err());
}
