use groggy::viz::embeddings::{
    spectral::{SpectralEmbedding, SpectralEmbeddingBuilder},
    EmbeddingEngine,
};
use groggy::{Graph, GraphMatrix};
fn full(graph: &Graph, normalized: bool, threshold: f64) -> GraphMatrix {
    SpectralEmbeddingBuilder::new()
        .normalized(normalized)
        .eigenvalue_threshold(threshold)
        .skip_constant_eigenvector(false)
        .compute(graph, graph.node_count())
        .unwrap()
}
fn check_eigenvectors(v: &GraphMatrix, l: &[[f64; 3]; 3]) -> Vec<f64> {
    let mut lambdas = Vec::new();
    for col in 0..v.shape().1 {
        let x: Vec<_> = (0..3).map(|row| v.get(row, col).unwrap()).collect();
        let lx: Vec<f64> = l
            .iter()
            .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
            .collect();
        let lambda: f64 = x.iter().zip(&lx).map(|(a, b)| a * b).sum();
        lambdas.push(lambda);
        for (a, b) in lx.iter().zip(&x) {
            assert!((a - lambda * b).abs() < 1e-9);
        }
        for prev in 0..=col {
            let dot: f64 = (0..3).map(|r| v.get(r, prev).unwrap() * x[r]).sum();
            assert!((dot - if prev == col { 1. } else { 0. }).abs() < 1e-9);
        }
    }
    for pair in lambdas.windows(2) {
        assert!(pair[0] <= pair[1] + 1e-9);
    }
    lambdas
}
#[test]
fn every_three_node_undirected_multigraph_matches_independent_laplacian() {
    let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    for mask in 0..64 {
        let mut graph = Graph::new_manual();
        graph.add_nodes(3).unwrap();
        let mut a = [[0.; 3]; 3];
        for (bit, &(u, v)) in pairs.iter().enumerate() {
            if mask & (1 << bit) == 0 {
                continue;
            }
            for _ in 0..=bit % 2 {
                graph.add_edge(u, v).unwrap();
                a[u][v] += 1.;
                if u != v {
                    a[v][u] += 1.;
                }
            }
        }
        let degrees: [f64; 3] = a.map(|row| row.iter().sum());
        let mut reach = [[false; 3]; 3];
        for (u, row) in reach.iter_mut().enumerate() {
            for (v, value) in row.iter_mut().enumerate() {
                *value = u == v || a[u][v] > 0.;
            }
        }
        for k in 0..3 {
            for u in 0..3 {
                for v in 0..3 {
                    reach[u][v] |= reach[u][k] && reach[k][v];
                }
            }
        }
        let mut representatives = Vec::new();
        for (u, row) in reach.iter().enumerate() {
            if !representatives.iter().any(|&v| row[v]) {
                representatives.push(u);
            }
        }
        for normalized in [false, true] {
            let mut l = [[0.; 3]; 3];
            for (u, row) in l.iter_mut().enumerate() {
                for (v, value) in row.iter_mut().enumerate() {
                    *value = if u == v { degrees[u] } else { 0. } - a[u][v];
                    if normalized {
                        *value = if degrees[u] > 0. && degrees[v] > 0. {
                            *value / degrees[u].sqrt() / degrees[v].sqrt()
                        } else {
                            0.
                        };
                    }
                }
            }
            let vectors = full(&graph, normalized, 1e-8);
            let lambda = check_eigenvectors(&vectors, &l);
            assert!(
                (lambda.iter().sum::<f64>() - (0..3).map(|i| l[i][i]).sum::<f64>()).abs() < 1e-9
            );
            // Selection threshold must not change the normalized operator.
            check_eigenvectors(&full(&graph, normalized, 10.), &l);
            let positive = 3 - representatives.len();
            let engine = SpectralEmbedding::new(normalized, 1e-300);
            if positive == 0 {
                assert!(engine.compute_embedding(&graph, 1).is_err());
                continue;
            }
            let selected = engine.compute_embedding(&graph, positive).unwrap();
            let values = check_eigenvectors(&selected, &l);
            assert!(
                values.iter().all(|&v| v > 1e-8),
                "mask={mask}, normalized={normalized}, values={values:?}"
            );
            for col in 0..positive {
                for &root in &representatives {
                    let dot: f64 = (0..3)
                        .filter(|&r| reach[root][r])
                        .map(|r| {
                            let null = if normalized && degrees[r] > 0. {
                                degrees[r].sqrt()
                            } else {
                                1.
                            };
                            selected.get(r, col).unwrap() * null
                        })
                        .sum();
                    assert!(dot.abs() < 1e-8, "mask={mask}, normalized={normalized}");
                }
            }
        }
    }
}
#[test]
fn sparse_active_rows_are_labeled_and_materialized_independently_of_graph() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(5).unwrap();
    graph.remove_node(1).unwrap();
    graph.remove_node(4).unwrap();
    graph.add_edge(0, 2).unwrap();
    graph.add_edge(0, 2).unwrap();
    graph.add_edge(2, 3).unwrap();
    graph.add_edge(2, 2).unwrap();
    let matrix = SpectralEmbedding::new(true, 1e-8)
        .compute_embedding(&graph, 2)
        .unwrap();
    let before = matrix.to_vec().unwrap();
    assert_eq!(matrix.row_labels().unwrap(), ["node_0", "node_2", "node_3"]);
    assert_eq!(matrix.column_names(), ["spectral_0", "spectral_1"]);
    graph.remove_node(2).unwrap();
    assert_eq!(matrix.to_vec().unwrap(), before);
    assert_eq!(matrix.shape(), (3, 2));
}
#[test]
fn invalid_domain_dimensions_and_nullspace_choices_are_explicit() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 1).unwrap();
    for threshold in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(SpectralEmbedding::new(true, threshold)
            .compute_embedding(&graph, 1)
            .is_err());
    }
    for dimensions in [0, 2, 3, 4] {
        assert!(SpectralEmbedding::default()
            .compute_embedding(&graph, dimensions)
            .is_err());
    }
    assert!(SpectralEmbedding::new(true, 10.)
        .compute_embedding(&graph, 1)
        .is_err());
    let mut directed = Graph::new_directed_manual();
    directed.add_nodes(2).unwrap();
    directed.add_edge(0, 1).unwrap();
    assert!(SpectralEmbedding::default()
        .compute_embedding(&directed, 1)
        .is_err());
    let empty = Graph::new_manual();
    assert!(SpectralEmbedding::default()
        .compute_embedding(&empty, 1)
        .is_err());
    let mut singleton = Graph::new_manual();
    singleton.add_node().unwrap();
    assert!(SpectralEmbedding::default()
        .compute_embedding(&singleton, 1)
        .is_err());
    let matrix = full(&singleton, true, 1e-8);
    assert_eq!(matrix.shape(), (1, 1));
    assert_eq!(matrix.get(0, 0), Some(1.));
}
