use groggy::{subgraphs::GraphView, types::GraphType, AttrValue as V, Graph};
use std::{collections::HashSet, sync::atomic::AtomicBool};
#[test]
fn paths_match_exhaustive_shortest_costs_and_actual_selected_edge_instances() {
    for directed in [false, true] {
        for mask in 0..512 {
            let mut graph = Graph::new_with_type_manual(if directed {
                GraphType::Directed
            } else {
                GraphType::Undirected
            });
            let all = graph.add_nodes(4).unwrap();
            graph.remove_node(all[1]).unwrap();
            let n = [all[0], all[2], all[3]];
            let mut distance = vec![vec![f64::INFINITY; 3]; 3];
            for (i, row) in distance.iter_mut().enumerate() {
                row[i] = 0.;
            }
            let mut records = Vec::new();
            for a in 0..3 {
                for b in 0..3 {
                    let bit = 3 * a + b;
                    if mask & (1 << bit) != 0 {
                        let w = (bit % 4) as f64;
                        for weight in [w + 5., w, w] {
                            let edge = graph.add_edge(n[a], n[b]).unwrap();
                            graph
                                .set_edge_attr(edge, "w".into(), V::Int(weight as i64))
                                .unwrap();
                            records.push((edge, n[a], n[b], weight));
                        }
                        distance[a][b] = distance[a][b].min(w);
                        if !directed {
                            distance[b][a] = distance[b][a].min(w);
                        }
                    }
                }
            }
            for k in 0..3 {
                for a in 0..3 {
                    for b in 0..3 {
                        distance[a][b] = distance[a][b].min(distance[a][k] + distance[k][b]);
                    }
                }
            }
            let input = graph.read_view().unwrap().algorithm_input().unwrap();
            for start in 0..3 {
                for goal in 0..3 {
                    let result = input.astar_path(n[start], n[goal], Some("w"), None, None);
                    if !distance[start][goal].is_finite() {
                        assert!(result.is_err());
                        continue;
                    }
                    let path = result.unwrap();
                    assert_eq!(path.total_weight, distance[start][goal]);
                    assert_eq!(path.nodes.first(), Some(&n[start]));
                    assert_eq!(path.nodes.last(), Some(&n[goal]));
                    assert_eq!(path.edges.len() + 1, path.nodes.len());
                    assert_eq!(path.metadata.length, path.edges.len());
                    assert_eq!(
                        path.nodes.iter().copied().collect::<HashSet<_>>().len(),
                        path.nodes.len()
                    );
                    let mut sum = 0.;
                    for (pair, &edge) in path.nodes.windows(2).zip(&path.edges) {
                        let record = records.iter().find(|&&(id, _, _, _)| id == edge).unwrap();
                        assert!(
                            (record.1 == pair[0] && record.2 == pair[1])
                                || (!directed && record.2 == pair[0] && record.1 == pair[1])
                        );
                        sum += record.3;
                        let expected = records
                            .iter()
                            .filter(|&&(_, s, t, w)| {
                                w == record.3
                                    && ((s == pair[0] && t == pair[1])
                                        || (!directed && t == pair[0] && s == pair[1]))
                            })
                            .map(|&(id, _, _, _)| id)
                            .min()
                            .unwrap();
                        assert_eq!(edge, expected);
                    }
                    assert_eq!(sum, path.total_weight);
                }
            }
        }
    }
}
#[test]
fn heuristic_domain_and_scope_are_checked_before_trivial_or_unreachable_results() {
    let mut graph = Graph::new_with_type_manual(GraphType::Directed);
    let n = graph.add_nodes(4).unwrap();
    let edge = graph.add_edge(n[0], n[1]).unwrap();
    graph.set_edge_attr(edge, "w".into(), V::Int(2)).unwrap();
    graph.set_node_attr(n[0], "h".into(), V::Int(2)).unwrap();
    graph.set_node_attr(n[1], "h".into(), V::Int(0)).unwrap();
    let good = graph.read_view().unwrap();
    assert_eq!(
        good.astar_path(n[0], n[1], Some("w"), Some("h"))
            .unwrap()
            .edges,
        vec![edge]
    );
    for bad in [
        V::Null,
        V::Bool(true),
        V::Text("2".into()),
        V::Int(-1),
        V::Float(f32::NAN),
        V::Float(f32::INFINITY),
    ] {
        graph.set_node_attr(n[3], "h".into(), bad).unwrap();
        assert!(graph
            .read_view()
            .unwrap()
            .astar_path(n[0], n[0], Some("w"), Some("h"))
            .is_err());
    }
    graph.set_node_attr(n[3], "h".into(), V::Int(0)).unwrap();
    graph.set_node_attr(n[0], "h".into(), V::Int(3)).unwrap();
    assert!(graph
        .read_view()
        .unwrap()
        .astar_path(n[0], n[1], Some("w"), Some("h"))
        .is_err());
    assert!(good
        .astar_path(n[0], n[0], Some("w"), Some("h"))
        .unwrap_err()
        .to_string()
        .contains("goal must be zero"));
    assert!(good
        .select_nodes(&[n[0]])
        .unwrap()
        .astar_path(n[0], n[1], None, None)
        .is_err());
    assert!(good
        .astar_path(n[1], n[0], None, None)
        .unwrap_err()
        .to_string()
        .contains("unreachable"));
}
#[test]
fn frozen_parallel_path_keeps_weights_heuristics_and_edge_scope_after_owner_drop() {
    let mut graph = Graph::new_with_type_manual(GraphType::Directed);
    let n = graph.add_nodes(3).unwrap();
    let first = graph.add_edge(n[0], n[1]).unwrap();
    let second = graph.add_edge(n[1], n[2]).unwrap();
    let excluded = graph.add_edge(n[0], n[2]).unwrap();
    graph.set_edge_attr(first, "w".into(), V::Int(4)).unwrap();
    graph.set_edge_attr(second, "w".into(), V::Int(1)).unwrap();
    graph
        .set_edge_attr(excluded, "w".into(), V::Int(0))
        .unwrap();
    for (node, h) in [(n[0], 5), (n[1], 1), (n[2], 0)] {
        graph.set_node_attr(node, "h".into(), V::Int(h)).unwrap();
    }
    let root = graph.read_view().unwrap();
    let view = GraphView::from_membership(
        root.read_state().clone(),
        n.iter().copied().collect(),
        HashSet::from([first, second]),
    )
    .unwrap();
    graph.set_edge_attr(first, "w".into(), V::Int(999)).unwrap();
    graph.set_node_attr(n[0], "h".into(), V::Int(999)).unwrap();
    graph.remove_node(n[1]).unwrap();
    drop(graph);
    let input = view.algorithm_input().unwrap();
    let path = input
        .astar_path(n[0], n[2], Some("w"), Some("h"), None)
        .unwrap();
    assert_eq!(path.nodes, n);
    assert_eq!(path.edges, vec![first, second]);
    assert_eq!(path.total_weight, 5.);
    assert!(input
        .astar_path(
            n[0],
            n[2],
            Some("w"),
            Some("h"),
            Some(&AtomicBool::new(true))
        )
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
    let worker = std::thread::spawn(move || {
        input
            .astar_path(n[0], n[2], Some("w"), Some("h"), None)
            .unwrap()
    });
    assert_eq!(worker.join().unwrap().total_weight, 5.);
}
