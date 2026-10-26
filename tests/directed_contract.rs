use groggy::query::query::EdgeFilter;
use groggy::query::traversal::{PathFindingOptions, TraversalOptions};
use groggy::storage::matrix::GraphMatrix;
use groggy::types::AttrValue;
use groggy::Graph;

fn assert_oriented(matrix: &GraphMatrix<f64>, source: usize, target: usize, weight: f64) {
    let labels = matrix.column_names();
    let source_index = labels
        .iter()
        .position(|label| label == &format!("node_{source}"))
        .unwrap();
    let target_index = labels
        .iter()
        .position(|label| label == &format!("node_{target}"))
        .unwrap();
    assert_eq!(matrix.get(source_index, target_index), Some(weight));
    assert_eq!(matrix.get(target_index, source_index), Some(0.0));
}

#[test]
fn directed_paths_follow_only_outgoing_edges() {
    let mut graph = Graph::new_directed_manual();
    let source = graph.add_node().unwrap();
    let target = graph.add_node().unwrap();
    let edge = graph.add_edge(source, target).unwrap();

    assert!(graph.has_edge_between(source, target).unwrap());
    assert!(!graph.has_edge_between(target, source).unwrap());
    assert_eq!(
        graph
            .find_edges(EdgeFilter::ConnectsNodes { source, target })
            .unwrap(),
        vec![edge]
    );
    assert!(graph
        .find_edges(EdgeFilter::ConnectsNodes {
            source: target,
            target: source,
        })
        .unwrap()
        .is_empty());
    assert_eq!(graph.neighbors(source).unwrap(), vec![target]);
    assert!(graph.neighbors(target).unwrap().is_empty());
    assert_eq!(
        graph
            .bfs(target, TraversalOptions::default())
            .unwrap()
            .nodes,
        vec![target]
    );
    assert_eq!(
        graph
            .dfs(target, TraversalOptions::default())
            .unwrap()
            .nodes,
        vec![target]
    );

    let forward = graph
        .shortest_path(source, target, PathFindingOptions::default())
        .unwrap()
        .unwrap();
    assert_eq!(forward.nodes, vec![source, target]);
    assert_eq!(forward.edges, vec![edge]);
    assert!(graph
        .shortest_path(target, source, PathFindingOptions::default())
        .unwrap()
        .is_none());
    assert_eq!(graph.all_paths(source, target, 3).unwrap().len(), 1);
    assert!(graph.all_paths(target, source, 3).unwrap().is_empty());
}

#[test]
fn directed_adjacency_preserves_edge_orientation() {
    let mut graph = Graph::new_directed_manual();
    let source = graph.add_node().unwrap();
    let target = graph.add_node().unwrap();
    let edge = graph.add_edge(source, target).unwrap();
    graph
        .set_edge_attr(edge, "weight".into(), AttrValue::Float(2.0))
        .unwrap();

    let adjacency = graph.to_adjacency_matrix::<f64>().unwrap();
    assert_oriented(&adjacency, source, target, 1.0);

    let legacy = graph.adjacency_matrix().unwrap();
    assert_oriented(&legacy, source, target, 1.0);

    let weighted = graph.to_weighted_adjacency_matrix::<f64>("weight").unwrap();
    assert_oriented(&weighted, source, target, 2.0);

    let subgraph = graph.subgraph_adjacency_matrix(&[source, target]).unwrap();
    assert_oriented(&subgraph, source, target, 1.0);
}

#[test]
fn undirected_adjacency_keeps_existing_symmetric_behavior() {
    let mut graph = Graph::new_manual();
    let source = graph.add_node().unwrap();
    let target = graph.add_node().unwrap();
    graph.add_edge(source, target).unwrap();

    let adjacency = graph.to_adjacency_matrix::<f64>().unwrap();
    let labels = adjacency.column_names();
    let source_index = labels
        .iter()
        .position(|label| label == &format!("node_{source}"))
        .unwrap();
    let target_index = labels
        .iter()
        .position(|label| label == &format!("node_{target}"))
        .unwrap();
    assert_eq!(adjacency.get(source_index, target_index), Some(1.0));
    assert_eq!(adjacency.get(target_index, source_index), Some(1.0));
    assert!(graph
        .shortest_path(target, source, PathFindingOptions::default())
        .unwrap()
        .is_some());
}
