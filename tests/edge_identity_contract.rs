use groggy::query::traversal::{PathFindingOptions, TraversalOptions};
use groggy::types::AttrValue;
use groggy::Graph;

#[test]
fn parallel_edges_remain_distinct_through_paths_and_history() {
    let mut graph = Graph::new_directed_manual();
    let source = graph.add_node().unwrap();
    let target = graph.add_node().unwrap();
    let expensive = graph.add_edge(source, target).unwrap();
    let cheap = graph.add_edge(source, target).unwrap();
    graph
        .set_edge_attr(expensive, "weight".into(), AttrValue::Float(10.0))
        .unwrap();
    graph
        .set_edge_attr(cheap, "weight".into(), AttrValue::Float(1.0))
        .unwrap();

    assert_ne!(expensive, cheap);
    assert_eq!(graph.edge_count(), 2);
    assert_eq!(graph.incident_edges(source).unwrap().len(), 2);
    let mut paths = graph.all_paths(source, target, 3).unwrap();
    paths.sort_by_key(|path| path.edges[0]);
    assert_eq!(paths.len(), 2);
    assert_eq!(paths[0].nodes, vec![source, target]);
    assert_eq!(paths[0].edges, vec![expensive]);
    assert_eq!(paths[1].edges, vec![cheap]);

    let shortest = graph
        .shortest_path(
            source,
            target,
            PathFindingOptions {
                weight_attribute: Some("weight".into()),
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
    assert_eq!(shortest.edges, vec![cheap]);
    assert_eq!(shortest.total_weight, 1.0);

    let state = graph
        .commit("parallel facts".into(), "test".into())
        .unwrap();
    let snapshot = graph.snapshot_at_commit(state).unwrap();
    assert!(snapshot.edge_exists(expensive));
    assert!(snapshot.edge_exists(cheap));
    assert_eq!(
        snapshot.edge_attr(expensive, &"weight".into()),
        Some(AttrValue::Float(10.0))
    );
    assert_eq!(
        snapshot.edge_attr(cheap, &"weight".into()),
        Some(AttrValue::Float(1.0))
    );
}

#[test]
fn self_loop_has_one_stable_edge_identity() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    let loop_edge = graph.add_edge(node, node).unwrap();

    assert_eq!(graph.edge_ids(), vec![loop_edge]);
    assert_eq!(graph.edge_endpoints(loop_edge).unwrap(), (node, node));
    assert_eq!(graph.incident_edges(node).unwrap(), vec![loop_edge]);
    assert_eq!(graph.degree(node).unwrap(), 1);
    assert_eq!(
        graph.bfs(node, TraversalOptions::default()).unwrap().nodes,
        vec![node]
    );
}
