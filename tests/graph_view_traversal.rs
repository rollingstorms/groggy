use groggy::{algorithms::AlgorithmRead, subgraphs::GraphView, types::GraphType, AttrValue, Graph};
use std::{
    collections::HashSet,
    sync::{atomic::AtomicBool, Arc},
};

fn expected_dfs(
    node: usize,
    adjacency: &[Vec<usize>],
    visited: &mut [bool],
    output: &mut Vec<usize>,
) {
    visited[node] = true;
    output.push(node);
    for &next in &adjacency[node] {
        if !visited[next] {
            expected_dfs(next, adjacency, visited, output);
        }
    }
}
#[test]
fn frozen_traversals_match_exhaustive_small_topologies_with_sparse_ids() {
    for directed in [false, true] {
        for mask in 0..512 {
            let mut graph = Graph::new_with_type(if directed {
                GraphType::Directed
            } else {
                GraphType::Undirected
            });
            let allocated = graph.add_nodes(4);
            graph.remove_node(allocated[1]).unwrap();
            let ids = [allocated[0], allocated[2], allocated[3]];
            let mut adjacency = vec![Vec::new(); 3];
            let mut distance = vec![vec![100usize; 3]; 3];
            for (i, row) in distance.iter_mut().enumerate() {
                row[i] = 0;
            }
            for a in 0..3 {
                for b in 0..3 {
                    if mask & (1 << (3 * a + b)) != 0 {
                        graph.add_edge(ids[a], ids[b]).unwrap();
                        // Duplicate an instance without changing reachability/order.
                        if a == 0 && b == 1 {
                            graph.add_edge(ids[a], ids[b]).unwrap();
                        }
                        adjacency[a].push(b);
                        distance[a][b] = distance[a][b].min(1);
                        if !directed {
                            adjacency[b].push(a);
                            distance[b][a] = distance[b][a].min(1);
                        }
                    }
                }
            }
            for row in &mut adjacency {
                row.sort_unstable();
                row.dedup();
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
                let expected: Vec<_> = (0..3)
                    .filter(|&i| distance[start][i] < 100)
                    .map(|i| (ids[i], distance[start][i] as i64))
                    .collect();
                assert_eq!(
                    input.bfs_distances(ids[start], None).unwrap(),
                    expected,
                    "BFS directed={directed} mask={mask} start={start}"
                );
                let mut visit = Vec::new();
                expected_dfs(start, &adjacency, &mut [false; 3], &mut visit);
                let expected: Vec<_> = visit
                    .iter()
                    .enumerate()
                    .map(|(order, &row)| (ids[row], order as i64))
                    .collect();
                assert_eq!(
                    input.dfs_order(ids[start], None).unwrap(),
                    expected,
                    "DFS directed={directed} mask={mask} start={start}"
                );
            }
        }
    }
}

#[test]
fn prepared_reads_share_pins_rows_and_csr_across_mutations_owner_drop_and_threads() {
    let mut graph = Graph::new_with_type(GraphType::Directed);
    let n = graph.add_nodes(5);
    let first = graph.add_edge(n[0], n[2]).unwrap();
    let onward = graph.add_edge(n[2], n[1]).unwrap();
    graph.add_edge(n[0], n[1]).unwrap(); // Excluded shortcut changes the distance if leaked.
    graph.add_edge(n[1], n[3]).unwrap(); // Excluded outside connection.
    let parallel = graph.add_edge(n[0], n[2]).unwrap();
    let loop_edge = graph.add_edge(n[2], n[2]).unwrap();
    graph
        .set_node_attr(n[0], "score".into(), AttrValue::Int(7))
        .unwrap();
    let root = graph.read_view().unwrap();
    let view = GraphView::from_membership(
        root.read_state().clone(),
        n.iter().copied().collect(),
        HashSet::from([first, onward, parallel, loop_edge]),
    )
    .unwrap();
    let input = AlgorithmRead::new(view.clone()).unwrap();
    let clone = input.clone();
    assert!(std::ptr::eq(input.csr(), clone.csr()));
    assert!(std::ptr::eq(
        input.node_ids().as_ptr(),
        clone.node_ids().as_ptr()
    ));
    assert_eq!(input.csr().neighbors(0), &[2, 2]);
    assert_eq!(input.csr().neighbors(2), &[1, 2]);
    assert_eq!(input.node_ids(), n);
    graph.remove_node(n[0]).unwrap();
    drop(graph);
    let expected = vec![(n[0], 0), (n[1], 2), (n[2], 1)];
    assert_eq!(input.bfs_distances(n[0], None).unwrap(), expected);
    assert_eq!(
        input.dfs_order(n[0], None).unwrap(),
        vec![(n[0], 0), (n[2], 1), (n[1], 2)]
    );
    assert_eq!(
        input.view().node_attr(n[0], &"score".into()).unwrap(),
        Some(AttrValue::Int(7))
    );
    let worker = std::thread::spawn(move || clone.bfs_distances(n[0], None).unwrap());
    assert_eq!(worker.join().unwrap(), expected);
    assert_eq!(view.bfs_distances(view.node_ids()[0]).unwrap(), expected);
}

#[test]
fn empty_isolated_out_of_scope_and_cancelled_inputs_have_explicit_results() {
    let empty = Graph::new().read_view().unwrap().algorithm_input().unwrap();
    assert_eq!(empty.csr().offsets, vec![0]);
    assert!(empty.bfs_distances(0, None).is_err());
    let mut graph = Graph::new();
    let n = graph.add_nodes(2);
    let view = graph.read_view().unwrap().select_nodes(&[n[1]]).unwrap();
    assert_eq!(view.bfs_distances(n[1]).unwrap(), vec![(n[1], 0)]);
    assert_eq!(view.dfs_order(n[1]).unwrap(), vec![(n[1], 0)]);
    assert!(view.bfs_distances(n[0]).is_err());
    assert!(view.dfs_order(usize::MAX).is_err());
    let input = view.algorithm_input().unwrap();
    let cancelled = AtomicBool::new(true);
    assert!(input
        .bfs_distances(n[1], Some(&cancelled))
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
    assert!(input
        .dfs_order(n[1], Some(&cancelled))
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
}

#[test]
fn long_chain_dfs_is_iterative_and_reaches_exact_integer_rows() {
    let mut graph = Graph::new_with_type(GraphType::Directed);
    let n = graph.add_nodes(10000);
    for pair in n.windows(2) {
        graph.add_edge(pair[0], pair[1]).unwrap();
    }
    let input = Arc::new(graph.read_view().unwrap().algorithm_input().unwrap());
    let rows = input.dfs_order(n[0], None).unwrap();
    assert_eq!(rows.len(), n.len());
    assert_eq!(rows.last(), Some(&(n[n.len() - 1], 9999)));
}

#[test]
fn historical_recorded_state_uses_its_own_topology_after_owner_drop() {
    let mut graph = Graph::new_with_type(GraphType::Directed);
    let n = graph.add_nodes(3);
    graph.add_edge(n[0], n[1]).unwrap();
    let checkpoint = graph
        .commit("before extension".into(), "test".into())
        .unwrap();
    graph.add_edge(n[1], n[2]).unwrap();
    graph.commit("extended".into(), "test".into()).unwrap();
    let historical = graph.snapshot_at_commit(checkpoint).unwrap();
    let input = GraphView::all(historical.read_state().clone())
        .algorithm_input()
        .unwrap();
    assert_eq!(
        graph.read_view().unwrap().bfs_distances(n[0]).unwrap(),
        vec![(n[0], 0), (n[1], 1), (n[2], 2)]
    );
    drop(graph);
    assert_eq!(
        input.bfs_distances(n[0], None).unwrap(),
        vec![(n[0], 0), (n[1], 1)]
    );
    assert_eq!(input.view().node_count(), 3);
}
