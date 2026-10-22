use groggy::storage::GraphMatrix;

fn matrix(a: &[f64], rows: usize, cols: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(a.to_vec(), rows, cols, None).unwrap()
}
fn orthogonal(q: &GraphMatrix<f64>) {
    let (m, n) = q.shape();
    assert_eq!(m, n);
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..m)
                .map(|k| q.get(k, i).unwrap() * q.get(k, j).unwrap())
                .sum();
            assert!(
                (dot - if i == j { 1. } else { 0. }).abs() < 1e-10,
                "orthogonality {i},{j}: {dot}"
            );
        }
    }
}
fn verify_svd(data: &[f64], rows: usize, cols: usize) {
    let a = matrix(data, rows, cols);
    let (u, s, vt) = a.svd().unwrap();
    assert_eq!(u.shape(), (rows, rows));
    assert_eq!(vt.shape(), (cols, cols));
    assert_eq!(s.len(), rows.min(cols));
    assert!(s.iter().all(|v| v.is_finite() && *v >= 0.));
    assert!(s.windows(2).all(|w| w[0] >= w[1]));
    orthogonal(&u);
    orthogonal(&vt);
    let scale = data.iter().fold(1e-300_f64, |v, x| v.max(x.abs()));
    for row in 0..rows {
        for col in 0..cols {
            let value: f64 = (0..s.len())
                .map(|k| u.get(row, k).unwrap() * (s[k] / scale) * vt.get(k, col).unwrap())
                .sum();
            assert!(
                (value - data[row * cols + col] / scale).abs() < 1e-10,
                "reconstruction {row},{col}: {value}"
            );
        }
    }
}
#[test]
fn svd_reconstructs_rectangular_zero_rank_deficient_and_empty_matrices() {
    for (a, m, n) in [
        (vec![1., 2., 3., 4., 5., 6.], 3, 2),
        (vec![1., 2., 3., 4., 5., 6.], 2, 3),
        (vec![1., 2., 2., 4.], 2, 2),
        (vec![0.; 12], 4, 3),
        (vec![1., -2., 3., -4.], 4, 1),
        (vec![1., -2., 3., -4.], 1, 4),
        (vec![], 0, 3),
        (vec![], 3, 0),
        (vec![], 0, 0),
    ] {
        verify_svd(&a, m, n);
    }
    let (_, s, _) = matrix(&[1., 2., 2., 4.], 2, 2).svd().unwrap();
    assert!((s[0] - 5.).abs() < 1e-12);
    assert!(s[1] < 1e-12);
}
#[test]
fn svd_retains_multiple_singular_values_and_handles_rescaling() {
    for scale in [1., 1e-200, 1e200] {
        verify_svd(&[3. * scale, 0., 0., 4. * scale], 2, 2);
        let (_, s, _) = matrix(&[3. * scale, 0., 0., 4. * scale], 2, 2)
            .svd()
            .unwrap();
        assert!((s[0] / scale - 4.).abs() < 1e-12);
        assert!((s[1] / scale - 3.).abs() < 1e-12);
    }
    let (_, s, _) = matrix(&[1., 0., 0., 1e-12], 2, 2).svd().unwrap();
    assert!((s[1] - 1e-12).abs() < 1e-25);
    let a =
        GraphMatrix::<f32>::from_row_major_data(vec![1., 2., 3., 4., 5., 6.], 3, 2, None).unwrap();
    let (u, s, vt) = a.svd().unwrap();
    for row in 0..3 {
        for col in 0..2 {
            let v: f64 = (0..2)
                .map(|k| u.get(row, k).unwrap() as f64 * s[k] * vt.get(k, col).unwrap() as f64)
                .sum();
            assert!((v - a.get(row, col).unwrap() as f64).abs() < 1e-5);
        }
    }
}
fn verify_eigen(data: &[f64], n: usize) -> Vec<f64> {
    let a = matrix(data, n, n);
    let (values, v) = a.eigenvalue_decomposition().unwrap();
    assert_eq!(values.len(), n);
    orthogonal(&v);
    assert!(values.windows(2).all(|x| x[0] >= x[1]));
    let scale = data.iter().fold(1e-300_f64, |v, x| v.max(x.abs()));
    for (col, &eigenvalue) in values.iter().enumerate().take(n) {
        for row in 0..n {
            let av: f64 = (0..n)
                .map(|k| data[row * n + k] / scale * v.get(k, col).unwrap())
                .sum();
            assert!((av - v.get(row, col).unwrap() * (eigenvalue / scale)).abs() < 1e-10);
        }
    }
    values
}
#[test]
fn symmetric_eigenpairs_have_small_residuals_and_orthogonal_vectors() {
    assert!(verify_eigen(&[], 0).is_empty());
    assert_eq!(verify_eigen(&[-7.], 1), vec![-7.]);
    for scale in [1., 1e-200, 1e200] {
        let values = verify_eigen(&[2. * scale, scale, scale, 2. * scale], 2);
        assert!((values[0] / scale - 3.).abs() < 1e-12);
        assert!((values[1] / scale - 1.).abs() < 1e-12);
    }
    verify_eigen(&[0., 2., 2., 0.], 2);
    verify_eigen(&[2., 1., 1., 1., 2., 1., 1., 1., 2.], 3);
    verify_eigen(&[0.; 16], 4);
}
#[test]
fn decompositions_reject_invalid_domains_instead_of_fabricating_answers() {
    for value in [f64::NAN, f64::INFINITY] {
        let a = matrix(&[value], 1, 1);
        assert!(a.svd().is_err());
        assert!(a.eigenvalue_decomposition().is_err());
    }
    assert!(matrix(&[1., 2., 3., 4.], 2, 2)
        .eigenvalue_decomposition()
        .is_err());
    assert!(matrix(&[1., 2.], 1, 2).eigenvalue_decomposition().is_err());
    let a = matrix(&[f64::MAX; 4], 2, 2);
    assert!(a.svd().is_err());
    assert!(a.eigenvalue_decomposition().is_err());
    let int = GraphMatrix::<i64>::from_row_major_data(vec![1, 0, 0, 1], 2, 2, None).unwrap();
    assert!(int.svd().is_err());
}
#[test]
fn deterministic_dense_fixtures_have_valid_full_decompositions() {
    for m in 1..8 {
        for n in 1..8 {
            let a: Vec<_> = (0..m * n)
                .map(|i| ((i * 13 + 7) % 23) as f64 - 11.)
                .collect();
            verify_svd(&a, m, n);
        }
    }
    for n in 2..9 {
        let a: Vec<_> = (0..n * n)
            .map(|i| {
                let r = i / n;
                let c = i % n;
                ((r.min(c) * 7 + r.max(c) * 11) % 19) as f64 - 9.
            })
            .collect();
        verify_eigen(&a, n);
    }
}

#[test]
fn pca_projection_preserves_distances_in_a_rotated_rank_two_plane() {
    use groggy::viz::projection::{algorithms::PCAProjection, ProjectionEngine};
    let data = [3., 3., 0., -3., -3., 0., 1., -1., 0., -1., 1., 0.];
    let a = matrix(&data, 4, 3);
    let positions = PCAProjection::new(true, false)
        .project_embedding(&a, &groggy::Graph::new())
        .unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let original =
                (data[i * 3] - data[j * 3]).powi(2) + (data[i * 3 + 1] - data[j * 3 + 1]).powi(2);
            let projected = (positions[i].x - positions[j].x).powi(2)
                + (positions[i].y - positions[j].y).powi(2);
            assert!((original - projected).abs() < 1e-10);
        }
    }
}

#[test]
fn spectral_embedding_uses_actual_nonconstant_laplacian_eigenvectors() {
    use groggy::viz::embeddings::{spectral::SpectralEmbedding, EmbeddingEngine};
    let mut graph = groggy::Graph::new();
    let allocated = graph.add_nodes(5);
    graph.remove_node(allocated[1]).unwrap();
    let nodes = [allocated[0], allocated[2], allocated[3], allocated[4]];
    graph
        .add_edges(&[
            (nodes[0], nodes[1]),
            (nodes[1], nodes[2]),
            (nodes[2], nodes[3]),
        ])
        .unwrap();
    let v = SpectralEmbedding::new(false, 1e-8)
        .compute_embedding(&graph, 2)
        .unwrap();
    assert_eq!(
        v.row_labels().unwrap(),
        &["node_0", "node_2", "node_3", "node_4"]
    );
    for (col, lambda) in [(0, 2. - 2f64.sqrt()), (1, 2.)] {
        assert!((0..4).map(|i| v.get(i, col).unwrap()).sum::<f64>().abs() < 1e-10);
        for row in 0..4 {
            let center = v.get(row, col).unwrap();
            let mut lap = if row == 0 || row == 3 {
                center
            } else {
                2. * center
            };
            if row > 0 {
                lap -= v.get(row - 1, col).unwrap();
            }
            if row < 3 {
                lap -= v.get(row + 1, col).unwrap();
            }
            assert!((lap - lambda * center).abs() < 1e-10);
        }
    }
}

#[test]
fn f32_symmetric_eigendecomposition_is_available_with_f64_accumulation() {
    let a = GraphMatrix::<f32>::from_row_major_data(vec![2., 1., 1., 2.], 2, 2, None).unwrap();
    let (values, v) = a.eigenvalue_decomposition().unwrap();
    assert!((values[0] - 3.).abs() < 1e-12 && (values[1] - 1.).abs() < 1e-12);
    for (col, &eigenvalue) in values.iter().enumerate().take(2) {
        for row in 0..2 {
            let actual: f64 = (0..2)
                .map(|k| a.get(row, k).unwrap() as f64 * v.get(k, col).unwrap() as f64)
                .sum();
            assert!((actual - eigenvalue * v.get(row, col).unwrap() as f64).abs() < 1e-6);
        }
    }
}
