use groggy::{
    entities::{Edge, EntityEdge, EntityNode, Node},
    traits::{EdgeOperations, GraphEntity, NodeOperations},
    AttrValue, Graph,
};
use std::{cell::RefCell, rc::Rc};
#[test]
fn compatibility_names_share_canonical_owner_and_keep_constructor_policies() {
    let graph = Rc::new(RefCell::new(Graph::new()));
    let a = graph.borrow_mut().add_node();
    let b = graph.borrow_mut().add_node();
    let edge = graph.borrow_mut().add_edge(a, b).unwrap();
    let node = Node::new(a, graph.clone()).unwrap();
    let legacy: groggy::storage::node::EntityNode = EntityNode::new(a, graph.clone());
    let current_edge = Edge::new(edge, graph.clone()).unwrap();
    let legacy_edge: groggy::storage::edge::EntityEdge = EntityEdge::new(edge, graph.clone());
    assert_eq!(node.entity_id(), legacy.entity_id());
    assert_eq!(current_edge.entity_id(), legacy_edge.entity_id());
    assert!(Rc::ptr_eq(&node.graph_ref(), &legacy.graph_ref()));
    assert!(Rc::ptr_eq(
        &current_edge.graph_ref(),
        &legacy_edge.graph_ref()
    ));
    assert_eq!(node.neighbors().unwrap(), legacy.neighbors().unwrap());
    assert_eq!(
        current_edge.endpoints().unwrap(),
        legacy_edge.endpoints().unwrap()
    );
    assert!(Node::new(usize::MAX, graph.clone()).is_err());
    assert!(Edge::new(usize::MAX, graph.clone()).is_err());
    let absent = EntityNode::new(usize::MAX, graph.clone());
    assert!(!absent.exists());
    assert!(absent.node_attributes().is_err());
    let absent = EntityEdge::new(usize::MAX, graph);
    assert!(!absent.exists());
    assert!(absent.endpoints().is_err());
}
#[test]
fn adapter_mutations_and_retained_branch_restoration_use_the_same_graph() {
    let graph = Rc::new(RefCell::new(Graph::new()));
    let a = graph.borrow_mut().add_node();
    let b = graph.borrow_mut().add_node();
    let edge = graph.borrow_mut().add_edge(a, b).unwrap();
    let node = Node::new(a, graph.clone()).unwrap();
    let legacy = EntityNode::new(a, graph.clone());
    let current_edge = Edge::new(edge, graph.clone()).unwrap();
    let legacy_edge = EntityEdge::new(edge, graph.clone());
    node.set_attribute("score".into(), AttrValue::Int(1))
        .unwrap();
    current_edge
        .set_attribute("weight".into(), AttrValue::Float(2.5))
        .unwrap();
    graph
        .borrow_mut()
        .commit("base".into(), "test".into())
        .unwrap();
    graph.borrow_mut().create_branch("base".into()).unwrap();
    legacy
        .set_attribute("score".into(), AttrValue::Int(2))
        .unwrap();
    legacy_edge.set_weight(3.5).unwrap();
    assert_eq!(
        node.get_attribute(&"score".into()).unwrap(),
        Some(AttrValue::Int(2))
    );
    assert_eq!(current_edge.weight().unwrap(), Some(3.5));
    graph
        .borrow_mut()
        .commit("updated".into(), "test".into())
        .unwrap();
    graph.borrow_mut().checkout_branch("base".into()).unwrap();
    assert_eq!(
        legacy.get_attribute(&"score".into()).unwrap(),
        Some(AttrValue::Int(1))
    );
    assert_eq!(current_edge.weight().unwrap(), Some(2.5));
    graph.borrow_mut().remove_node(a).unwrap();
    assert!(!node.is_active());
    assert!(!legacy.exists());
    assert!(!legacy_edge.exists());
    assert!(node.get_attribute(&"score".into()).is_err());
    assert!(legacy.node_attributes().is_err());
    assert!(current_edge.endpoints().is_err());
    assert!(legacy_edge.edge_attributes().is_err());
    assert_eq!(graph.borrow().pool().get_edge_endpoints(edge), Some((a, b)));
    assert!(Node::new(a, graph.clone()).is_err());
    assert!(Edge::new(edge, graph).is_err());
}
