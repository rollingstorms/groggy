use groggy::types::{AttrValue, GraphType};
use groggy::Graph;

#[test]
fn empty_graph_is_a_valid_construction_state() {
    let graph = Graph::new_manual();

    assert!(graph.is_empty());
    assert!(graph.is_undirected());
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
    assert!(graph.node_ids().is_empty());
    assert!(graph.edge_ids().is_empty());
}

#[test]
fn one_node_without_edges_is_the_minimal_graph() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();

    assert!(!graph.is_empty());
    assert_eq!(graph.node_count(), 1);
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.node_ids(), vec![node]);
    assert!(graph.edge_ids().is_empty());
    assert!(graph.contains_node(node));
    assert!(graph.neighbors(node).unwrap().is_empty());
    assert_eq!(graph.degree(node).unwrap(), 0);
    assert_eq!(
        graph.get_node_attr(node, &"entity_type".into()).unwrap(),
        Some(AttrValue::Text("base".to_string()))
    );
}

#[test]
fn direction_is_selected_when_the_graph_is_created() {
    let graph = Graph::new_with_type_manual(GraphType::Directed);

    assert!(graph.is_directed());
    assert!(!graph.is_undirected());
    assert!(graph.is_empty());
}
