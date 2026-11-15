//! Independent finite graph references across captured, sparse-ID topology.
use groggy::{
    algorithms::{community::ComponentMode, AlgorithmRead},
    AttrValue, BatchNodeRef as N, Graph, GraphBatch,
};
use std::collections::VecDeque;

#[test]
fn captured_paths_components_and_degrees_match_independent_finite_references() {
    let nodes = [0, 2, 3, 5, 6, 7];
    for directed in [false, true] {
        for bits in 0u64..128 {
            let mut graph = if directed {
                Graph::new_directed()
            } else {
                Graph::new()
            };
            graph
                .apply_batch(
                    GraphBatch {
                        add_nodes: 8,
                        effective_at: Some(0),
                        ..Default::default()
                    },
                    0,
                    "nodes".into(),
                    "test".into(),
                )
                .unwrap();
            let mut edges = Vec::new();
            for i in 0..5 {
                if bits & (1 << i) != 0 {
                    edges.push((nodes[i], nodes[(i + 1) % 5], (i % 3) as f64));
                }
            }
            if bits & 32 != 0 {
                edges.extend([(0, 2, 3.0), (0, 2, 0.5)]);
            }
            if bits & 64 != 0 {
                edges.extend([(3, 3, 0.0), (5, 2, 2.0)]);
            }
            graph
                .apply_batch(
                    GraphBatch {
                        add_edges: edges
                            .iter()
                            .map(|&(u, v, _)| (N::Existing(u), N::Existing(v)))
                            .collect(),
                        remove_nodes: vec![N::Existing(1), N::Existing(4)],
                        edge_attrs: [(
                            "w".into(),
                            edges
                                .iter()
                                .enumerate()
                                .map(|(i, &(_, _, w))| {
                                    (groggy::BatchEdgeRef::Added(i), AttrValue::Float64(w))
                                })
                                .collect(),
                        )]
                        .into(),
                        effective_at: Some(2),
                        ..Default::default()
                    },
                    graph.current_state_id(),
                    "fixture".into(),
                    "test".into(),
                )
                .unwrap();
            let captured = graph.query().unwrap();
            graph.remove_node(0).unwrap();
            let read = AlgorithmRead::new(captured.clone()).unwrap();
            assert_eq!(read.node_ids(), nodes);
            let mut adjacency = vec![Vec::new(); nodes.len()];
            let mut distance = vec![vec![f64::INFINITY; nodes.len()]; nodes.len()];
            for (i, row) in distance.iter_mut().enumerate() {
                row[i] = 0.0;
            }
            for &(u, v, w) in &edges {
                let u = nodes.iter().position(|&id| id == u).unwrap();
                let v = nodes.iter().position(|&id| id == v).unwrap();
                adjacency[u].push(v);
                distance[u][v] = distance[u][v].min(w);
                if !directed {
                    adjacency[v].push(u);
                    distance[v][u] = distance[v][u].min(w);
                }
            }
            for k in 0..nodes.len() {
                for i in 0..nodes.len() {
                    for j in 0..nodes.len() {
                        distance[i][j] = distance[i][j].min(distance[i][k] + distance[k][j]);
                    }
                }
            }
            for (start, &id) in nodes.iter().enumerate() {
                let mut hops = vec![-1; nodes.len()];
                hops[start] = 0;
                let mut queue = VecDeque::from([start]);
                while let Some(u) = queue.pop_front() {
                    for &v in &adjacency[u] {
                        if hops[v] < 0 {
                            hops[v] = hops[u] + 1;
                            queue.push_back(v);
                        }
                    }
                }
                assert_eq!(
                    read.bfs_distances(id, None).unwrap(),
                    nodes
                        .iter()
                        .copied()
                        .zip(hops)
                        .filter(|(_, distance)| *distance >= 0)
                        .collect::<Vec<_>>()
                );
                let actual = read.dijkstra_distances(id, Some("w"), None).unwrap();
                let expected: Vec<_> = nodes
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(j, node)| {
                        distance[start][j]
                            .is_finite()
                            .then_some((node, distance[start][j]))
                    })
                    .collect();
                assert_eq!(actual, expected);
                let incident = edges
                    .iter()
                    .filter(|&&(u, v, _)| u == id || v == id)
                    .count();
                assert_eq!(captured.degree(id).unwrap(), incident);
                assert_eq!(
                    captured.in_degree(id).unwrap(),
                    if directed {
                        edges.iter().filter(|&&(_, v, _)| v == id).count()
                    } else {
                        incident
                    }
                );
                assert_eq!(
                    captured.out_degree(id).unwrap(),
                    if directed {
                        edges.iter().filter(|&&(u, _, _)| u == id).count()
                    } else {
                        incident
                    }
                );
            }
            let mode = if directed {
                ComponentMode::Strong
            } else {
                ComponentMode::Undirected
            };
            let components = read.connected_components(mode, None).unwrap();
            for (i, &u) in nodes.iter().enumerate() {
                for (j, &v) in nodes.iter().enumerate() {
                    let same = components
                        .iter()
                        .any(|c| c.contains_node(u) && c.contains_node(v));
                    assert_eq!(
                        same,
                        distance[i][j].is_finite() && distance[j][i].is_finite()
                    );
                }
            }
        }
    }
}
