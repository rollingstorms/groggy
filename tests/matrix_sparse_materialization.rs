use groggy::storage::GraphMatrix;
#[test]
fn dense_consumers_reject_large_implicit_csr_materialization() {
    let n = 4096;
    let a = GraphMatrix::from_csr(n, n, vec![1.; n], (0..n).collect(), (0..=n).collect()).unwrap();
    for error in [
        a.svd().unwrap_err(),
        a.svd_thin().unwrap_err(),
        a.qr_decomposition().unwrap_err(),
        a.eigenvalue_decomposition().unwrap_err(),
        a.rank().unwrap_err(),
        a.condition_number().unwrap_err(),
    ] {
        assert!(error.to_string().contains("to_dense()"), "{error}");
    }
    // Sparse-native consumers do not materialize the coefficient matrix.
    let rhs = GraphMatrix::ones(n, 1);
    let solution = a.solve(&rhs).unwrap();
    assert_eq!(solution.get(n - 1, 0), Some(1.));
    assert!(a.cholesky_decomposition().unwrap().is_sparse());
}
#[test]
fn small_sparse_dense_work_retains_existing_results_and_explicit_opt_in() {
    let a = GraphMatrix::from_csr(2, 2, vec![2., 3.], vec![0, 1], vec![0, 1, 2]).unwrap();
    assert_eq!(a.svd_thin().unwrap().1, vec![3., 2.]);
    let dense = a.to_dense().unwrap();
    assert_eq!(dense.eigenvalue_decomposition().unwrap().0, vec![3., 2.]);
    assert_eq!(GraphMatrix::<f64>::IMPLICIT_DENSE_ELEMENT_LIMIT, 1_048_576);
}
