//! Unsupported group attributes must not pretend to store or retrieve values.
use groggy::traits::GraphEntity;
use groggy::{AttrValue as V, Graph, GraphError};
use std::collections::HashSet;
#[test]
fn group_attribute_operations_fail_explicitly_without_mutation_or_allocation() {
    let mut graph = Graph::new();
    graph.add_nodes(2).unwrap();
    let source = graph.read_view().unwrap();
    let group = graph.retain_group(&source, "source".into()).unwrap();
    let selected = groggy::subgraphs::Subgraph::from_nodes(
        graph.into_shared(),
        HashSet::from([0, 1]),
        "selected".into(),
    )
    .unwrap();
    let owner = selected.graph_ref();
    let before = owner.borrow().current_state_id();
    let columns = owner.borrow().pool().attribute_names();
    assert!(matches!(
        selected.set_attribute("lost".into(), V::Int(7)),
        Err(GraphError::NotImplemented { .. })
    ));
    assert!(matches!(
        selected.get_attribute(&"lost".into()),
        Err(GraphError::NotImplemented { .. })
    ));
    let mut pool = groggy::storage::pool::GraphPool::new();
    assert!(matches!(
        pool.set_subgraph_attribute(group, "lost".into(), V::Null),
        Err(GraphError::NotImplemented { .. })
    ));
    assert!(matches!(
        pool.get_subgraph_attribute(group, &"lost".into()),
        Err(GraphError::NotImplemented { .. })
    ));
    assert_eq!(owner.borrow().current_state_id(), before);
    assert_eq!(owner.borrow().pool().attribute_names(), columns);
    assert!(!owner.borrow().has_uncommitted_changes());
}
#[test]
fn meta_and_subnode_property_edits_publish_independently_of_the_frozen_source() {
    let mut graph = Graph::new();
    let base = graph.add_node().unwrap();
    graph.set_node_attr(base, "v".into(), V::Int(1)).unwrap();
    let source = graph.read_view().unwrap();
    let group = graph.retain_group(&source, "source".into()).unwrap();
    let meta = graph.create_meta_node(group).unwrap();
    let owner = graph.into_shared();
    let node = groggy::entities::Node::new(base, owner.clone()).unwrap();
    let meta_node = groggy::entities::MetaNode::new(meta, owner.clone()).unwrap();
    let before = owner.borrow().current_state_id();
    meta_node
        .set_attribute("title".into(), V::Text("group".into()))
        .unwrap();
    node.set_attribute("v".into(), V::Int(2)).unwrap();
    let graph = owner.borrow();
    assert_eq!(graph.current_state_id(), before + 2);
    assert_eq!(
        graph.get_node_attr(meta, &"title".into()).unwrap(),
        Some(V::Text("group".into()))
    );
    assert_eq!(
        graph.get_node_attr(base, &"v".into()).unwrap(),
        Some(V::Int(2))
    );
    assert_eq!(
        graph
            .read_view()
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .node_attr(base, &"v".into())
            .unwrap(),
        Some(V::Int(1))
    );
    assert!(!graph.has_uncommitted_changes());
}
