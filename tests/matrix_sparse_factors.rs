use groggy::storage::GraphMatrix;
fn csr(a: Vec<f64>, n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(a, n, n, None)
        .unwrap()
        .to_csr()
        .unwrap()
}
#[test]
fn sparse_factors_solve_and_metadata_match_reconstruction_across_scales() {
    for scale in [1e-250, 1., 1e250] {
        let a = csr(vec![0., 2. * scale, scale, 3. * scale], 2);
        let (p, l, u) = a.lu_decomposition().unwrap();
        assert!(p.is_sparse() && l.is_sparse() && u.is_sparse());
        let expected = p.multiply(&a).unwrap();
        let actual = l.multiply(&u).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (actual.get(r, c).unwrap() / scale - expected.get(r, c).unwrap() / scale).abs()
                        < 1e-13
                );
            }
        }
        let b = GraphMatrix::from_row_major_data(vec![4. * scale, 7. * scale], 2, 1, None).unwrap();
        let x = a.solve(&b).unwrap();
        assert!((x.get(0, 0).unwrap() - 1.).abs() < 1e-13);
        assert!((x.get(1, 0).unwrap() - 2.).abs() < 1e-13);
        let a = csr(vec![4. * scale, scale, scale, 3. * scale], 2);
        let l = a.cholesky_decomposition().unwrap();
        assert!(l.is_sparse());
        let actual = l.multiply(&l.transpose().unwrap()).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (actual.get(r, c).unwrap() / scale - a.get(r, c).unwrap() / scale).abs()
                        < 1e-13
                );
            }
        }
    }
}
#[test]
fn sparse_factor_domains_singular_lu_and_large_diagonal_storage_are_explicit() {
    let a = csr(vec![1., 2., 2., 4.], 2);
    assert!(a.lu_decomposition().is_ok());
    assert!(a.solve(&GraphMatrix::ones(2, 1)).is_err());
    assert!(a.cholesky_decomposition().is_err());
    assert!(csr(vec![1., 2., 0., 1.], 2)
        .cholesky_decomposition()
        .is_err());
    assert!(csr(vec![f64::NAN], 1).lu_decomposition().is_err());
    let n = 10000;
    let a = GraphMatrix::from_csr(n, n, vec![4.; n], (0..n).collect(), (0..=n).collect()).unwrap();
    let (p, l, u) = a.lu_decomposition().unwrap();
    assert!(p.is_sparse() && l.is_sparse() && u.is_sparse());
    assert_eq!(u.zero_count().unwrap(), n * n - n);
    let l = a.cholesky_decomposition().unwrap();
    assert!(l.is_sparse());
    assert_eq!(l.get(n - 1, n - 1), Some(2.));
    let x = a.solve(&GraphMatrix::ones(n, 1)).unwrap();
    assert_eq!(x.get(n - 1, 0), Some(0.25));
    let a = GraphMatrix::<f32>::from_csr(1, 1, vec![4.], vec![0], vec![0, 1]).unwrap();
    assert_eq!(a.cholesky_decomposition().unwrap().get(0, 0), Some(2.));
    let a = GraphMatrix::<i64>::from_csr(1, 1, vec![4], vec![0], vec![0, 1]).unwrap();
    assert!(a.lu_decomposition().is_err());
}

#[test]
fn sparse_lu_and_solve_retain_independent_rows_across_extreme_dynamic_range() {
    let a = csr(vec![1e-300, 0., 0., 1e300], 2);
    let (_, _, u) = a.lu_decomposition().unwrap();
    assert_eq!(u.get(0, 0), Some(1e-300));
    assert_eq!(u.get(1, 1), Some(1e300));
    let b = GraphMatrix::from_row_major_data(vec![1e-300, 1e300], 2, 1, None).unwrap();
    let x = a.solve(&b).unwrap();
    assert_eq!(x.get(0, 0), Some(1.));
    assert_eq!(x.get(1, 0), Some(1.));
    let a = csr(vec![2., 0., 0., 0., 3., 0., 0., 0., 4.], 3);
    let (sign, log) = a.slogdet().unwrap();
    assert_eq!(sign, 1.);
    assert!((log - 24f64.ln()).abs() < 1e-14);
    assert!((a.determinant().unwrap() - 24.).abs() < 1e-12);
}
