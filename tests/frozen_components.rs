use groggy::{algorithms::community::ComponentMode, types::GraphType, Graph};
use std::sync::atomic::AtomicBool;

#[test]
fn frozen_components_match_reachability_and_preserve_selected_edge_instances() {
    for directed in [false, true] {
        for mask in 0..256 {
            let mut graph = Graph::new_with_type(if directed {
                GraphType::Directed
            } else {
                GraphType::Undirected
            });
            let nodes = graph.add_nodes(5);
            let pairs = [
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 3),
                (3, 2),
                (3, 0),
                (0, 0),
                (0, 1),
            ];
            let mut edges = Vec::new();
            let mut reach = [[false; 5]; 5];
            for (i, row) in reach.iter_mut().enumerate() {
                row[i] = true;
            }
            for (bit, &(s, t)) in pairs.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    edges.push(graph.add_edge(nodes[s], nodes[t]).unwrap());
                    reach[s][t] = true;
                    if !directed {
                        reach[t][s] = true;
                    }
                }
            }
            let view = graph.read_view().unwrap();
            graph.add_edge(nodes[0], nodes[4]).unwrap(); // Not captured.
            drop(graph);
            for mode in [
                ComponentMode::Undirected,
                ComponentMode::Weak,
                ComponentMode::Strong,
            ] {
                let mut paths = reach;
                if mode != ComponentMode::Strong {
                    for i in 0..5 {
                        for j in 0..5 {
                            if reach[i][j] {
                                paths[j][i] = true;
                            }
                        }
                    }
                }
                for k in 0..5 {
                    for i in 0..5 {
                        for j in 0..5 {
                            paths[i][j] |= paths[i][k] && paths[k][j];
                        }
                    }
                }
                let groups = view.connected_components(mode).unwrap();
                let mut actual = [usize::MAX; 5];
                for (label, group) in groups.iter().enumerate() {
                    for id in group.node_ids() {
                        actual[nodes.iter().position(|&n| n == id).unwrap()] = label;
                    }
                    let expected: Vec<_> = edges
                        .iter()
                        .copied()
                        .filter(|&edge| {
                            let (s, t) = view.edge_endpoints(edge).unwrap();
                            group.contains_node(s) && group.contains_node(t)
                        })
                        .collect();
                    assert_eq!(group.edge_ids(), expected);
                }
                for i in 0..5 {
                    for j in 0..5 {
                        assert_eq!(actual[i] == actual[j], paths[i][j] && paths[j][i]);
                    }
                }
                assert!(groups
                    .windows(2)
                    .all(|p| p[0].node_ids()[0] < p[1].node_ids()[0]));
            }
        }
    }
}

#[test]
fn components_handle_empty_isolates_selection_and_cancellation() {
    let mut graph = Graph::new();
    assert!(graph
        .read_view()
        .unwrap()
        .connected_components(ComponentMode::Strong)
        .unwrap()
        .is_empty());
    let n = graph.add_nodes(3);
    let edge = graph.add_edge(n[0], n[1]).unwrap();
    let excluded = graph.add_edge(n[1], n[2]).unwrap();
    let view = graph.read_view().unwrap().select_edges(&[edge]).unwrap();
    let groups = view.connected_components(ComponentMode::Weak).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].edge_ids(), vec![edge]);
    assert!(!groups[0].contains_edge(excluded));
    let cancelled = AtomicBool::new(true);
    assert!(view
        .algorithm_input()
        .unwrap()
        .connected_components(ComponentMode::Strong, Some(&cancelled))
        .is_err());
}
