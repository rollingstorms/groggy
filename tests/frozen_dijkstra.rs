use groggy::{subgraphs::GraphView, types::GraphType, AttrValue as V, Graph};
use std::{collections::HashSet, sync::atomic::AtomicBool};
#[test]
fn frozen_weighted_distances_match_floyd_warshall_on_small_multigraphs() {
    for directed in [false, true] {
        for mask in 0..512 {
            let mut graph = Graph::new_with_type(if directed {
                GraphType::Directed
            } else {
                GraphType::Undirected
            });
            let all = graph.add_nodes(4);
            graph.remove_node(all[1]).unwrap();
            let n = [all[0], all[2], all[3]];
            let mut d = vec![vec![f64::INFINITY; 3]; 3];
            for (i, row) in d.iter_mut().enumerate() {
                row[i] = 0.;
            }
            for a in 0..3 {
                for b in 0..3 {
                    let bit = 3 * a + b;
                    if mask & (1 << bit) != 0 {
                        let weight = (bit % 4) as f64;
                        let edge = graph.add_edge(n[a], n[b]).unwrap();
                        graph
                            .set_edge_attr(edge, "w".into(), V::Int(weight as i64))
                            .unwrap();
                        let parallel = graph.add_edge(n[a], n[b]).unwrap();
                        graph
                            .set_edge_attr(parallel, "w".into(), V::Int(weight as i64 + 7))
                            .unwrap();
                        d[a][b] = d[a][b].min(weight);
                        if !directed {
                            d[b][a] = d[b][a].min(weight);
                        }
                    }
                }
            }
            for k in 0..3 {
                for a in 0..3 {
                    for b in 0..3 {
                        d[a][b] = d[a][b].min(d[a][k] + d[k][b]);
                    }
                }
            }
            let input = graph.read_view().unwrap().algorithm_input().unwrap();
            for start in 0..3 {
                let expected: Vec<_> = (0..3)
                    .filter(|&i| d[start][i].is_finite())
                    .map(|i| (n[i], d[start][i]))
                    .collect();
                assert_eq!(
                    input.dijkstra_distances(n[start], Some("w"), None).unwrap(),
                    expected,
                    "directed={directed} mask={mask} start={start}"
                );
            }
        }
    }
}
#[test]
fn captured_weights_and_selected_parallel_instances_survive_mutation_and_owner_drop() {
    let mut graph = Graph::new_with_type(GraphType::Directed);
    let n = graph.add_nodes(4);
    let a = graph.add_edge(n[0], n[1]).unwrap();
    let b = graph.add_edge(n[1], n[2]).unwrap();
    let excluded = graph.add_edge(n[0], n[2]).unwrap();
    graph.set_edge_attr(a, "w".into(), V::Int(4)).unwrap();
    graph.set_edge_attr(b, "w".into(), V::Float(0.5)).unwrap();
    graph
        .set_edge_attr(excluded, "w".into(), V::Int(0))
        .unwrap();
    let root = graph.read_view().unwrap();
    let view = GraphView::from_membership(
        root.read_state().clone(),
        n.iter().copied().collect(),
        HashSet::from([a, b]),
    )
    .unwrap();
    graph.set_edge_attr(a, "w".into(), V::Int(999)).unwrap();
    graph.remove_node(n[1]).unwrap();
    drop(graph);
    assert_eq!(
        view.dijkstra_distances(n[0], Some("w")).unwrap(),
        vec![(n[0], 0.), (n[1], 4.), (n[2], 4.5)]
    );
    assert_eq!(
        view.dijkstra_distances(n[3], Some("w")).unwrap(),
        vec![(n[3], 0.)]
    );
    assert!(view
        .select_nodes(&[n[3]])
        .unwrap()
        .dijkstra_distances(n[0], None)
        .is_err());
    let input = view.algorithm_input().unwrap();
    let cancelled = AtomicBool::new(true);
    assert!(input
        .dijkstra_distances(n[0], Some("w"), Some(&cancelled))
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
    let start = n[0];
    let worker =
        std::thread::spawn(move || input.dijkstra_distances(start, Some("w"), None).unwrap());
    assert_eq!(
        worker.join().unwrap(),
        vec![(n[0], 0.), (n[1], 4.), (n[2], 4.5)]
    );
}
#[test]
fn selected_weights_are_validated_even_on_unreachable_edges_and_missing_is_one() {
    for bad in [
        V::Null,
        V::Bool(true),
        V::Text("2".into()),
        V::Float(f32::NAN),
        V::Float(f32::INFINITY),
        V::Int(-1),
    ] {
        let mut graph = Graph::new();
        let n = graph.add_nodes(3);
        let edge = graph.add_edge(n[1], n[2]).unwrap();
        graph.set_edge_attr(edge, "w".into(), bad).unwrap();
        let view = graph.read_view().unwrap();
        assert!(view.dijkstra_distances(n[0], Some("w")).is_err());
        assert_eq!(
            view.dijkstra_distances(n[0], None).unwrap(),
            vec![(n[0], 0.)]
        );
    }
    let mut graph = Graph::new();
    let n = graph.add_nodes(3);
    let a = graph.add_edge(n[0], n[1]).unwrap();
    graph.add_edge(n[1], n[2]).unwrap();
    graph.set_edge_attr(a, "w".into(), V::Int(0)).unwrap();
    assert_eq!(
        graph
            .read_view()
            .unwrap()
            .dijkstra_distances(n[0], Some("w"))
            .unwrap(),
        vec![(n[0], 0.), (n[1], 0.), (n[2], 1.)]
    );
}
