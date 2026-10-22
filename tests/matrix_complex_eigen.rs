use groggy::storage::GraphMatrix;
fn matrix(a: Vec<f64>, n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(a, n, n, None).unwrap()
}
#[test]
fn rotation_real_triangular_repeated_and_empty_pairs_reconstruct() {
    for a in [
        vec![0., -1., 1., 0.],
        vec![1., 3., 0., 2.],
        vec![2., 0., 0., 2.],
        vec![0.; 4],
    ] {
        for scale in [1e-250, 1., 1e250] {
            let input = matrix(a.iter().map(|v| v * scale).collect(), 2);
            let result = input.complex_eigenvalue_decomposition().unwrap();
            for col in 0..2 {
                let mut norm = 0.;
                for row in 0..2 {
                    let real = result.eigenvectors_real.get(row, col).unwrap();
                    let imag = result.eigenvectors_imag.get(row, col).unwrap();
                    norm += real * real + imag * imag;
                    let ar: f64 = (0..2)
                        .map(|k| a[row * 2 + k] * result.eigenvectors_real.get(k, col).unwrap())
                        .sum();
                    let ai: f64 = (0..2)
                        .map(|k| a[row * 2 + k] * result.eigenvectors_imag.get(k, col).unwrap())
                        .sum();
                    let lr = result.eigenvalues_real[col] / scale;
                    let li = result.eigenvalues_imag[col] / scale;
                    assert!((ar - lr * real + li * imag).abs() < 1e-11);
                    assert!((ai - lr * imag - li * real).abs() < 1e-11);
                }
                assert!((norm - 1.).abs() < 1e-12);
            }
            assert_eq!(
                result.eigenvectors_real.row_labels(),
                Some(input.column_names())
            );
        }
    }
    let empty = matrix(vec![], 0)
        .complex_eigenvalue_decomposition()
        .unwrap();
    assert!(empty.eigenvalues_real.is_empty());
    assert_eq!(empty.eigenvectors_imag.shape(), (0, 0));
}
#[test]
fn rejects_defective_nonfinite_nonsquare_and_integer_domains() {
    assert!(matrix(vec![1., 1., 0., 1.], 2)
        .complex_eigenvalue_decomposition()
        .is_err());
    for value in [f64::INFINITY, f64::NAN] {
        assert!(matrix(vec![value], 1)
            .complex_eigenvalue_decomposition()
            .is_err());
    }
    let a = GraphMatrix::<f64>::from_row_major_data(vec![1., 2.], 1, 2, None).unwrap();
    assert!(a.complex_eigenvalue_decomposition().is_err());
    let a = GraphMatrix::<i64>::from_row_major_data(vec![1], 1, 1, None).unwrap();
    assert!(a.complex_eigenvalue_decomposition().is_err());
    let a = GraphMatrix::<f32>::from_row_major_data(vec![0., -1., 1., 0.], 2, 2, None).unwrap();
    assert!(a.complex_eigenvalue_decomposition().is_ok());
}
