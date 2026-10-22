use groggy::storage::GraphMatrix;
fn matrix(v: Vec<f64>, m: usize, n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(v, m, n, None).unwrap()
}
#[test]
fn spectral_diagonal_derivatives_and_zero_axes_are_explicit() {
    let a = matrix(vec![3., 0., 0., 1.], 2, 2);
    let h = matrix(vec![2., 4., 4., 5.], 2, 2);
    let (ds, dv) = a.eigen_jvp(&h).unwrap();
    assert_eq!(ds, vec![2., 5.]);
    assert_eq!(dv.get(1, 0), Some(2.));
    assert_eq!(dv.get(0, 1), Some(-2.));
    let g = a.eigen_vjp(&[7., 9.], &matrix(vec![0.; 4], 2, 2)).unwrap();
    assert_eq!(g.get(0, 0), Some(7.));
    assert_eq!(g.get(1, 1), Some(9.));
    let (du, s, dvt) = a.svd_jvp(&h).unwrap();
    assert_eq!(s, vec![2., 5.]);
    assert_eq!(du.shape(), (2, 2));
    assert_eq!(dvt.shape(), (2, 2));
    for (m, n) in [(0, 0), (0, 3), (3, 0)] {
        let a = matrix(vec![], m, n);
        let (du, s, vt) = a.svd_jvp(&a).unwrap();
        assert_eq!(du.shape(), (m, 0));
        assert!(s.is_empty());
        assert_eq!(vt.shape(), (0, n));
        assert_eq!(a.svd_vjp(&du, &s, &vt).unwrap().shape(), (m, n));
    }
}
#[test]
fn spectral_derivative_domains_reject_degenerate_or_incompatible_inputs() {
    let identity = matrix(vec![1., 0., 0., 1.], 2, 2);
    assert!(identity.eigen_jvp(&identity).is_err());
    assert!(identity.svd_jvp(&identity).is_err());
    let a = matrix(vec![3., 0., 0., 0.], 2, 2);
    assert!(a.svd_jvp(&a).is_err());
    assert!(a.eigen_jvp(&matrix(vec![1., 1., 0., 1.], 2, 2)).is_err());
    assert!(a.eigen_vjp(&[1.], &a).is_err());
    assert!(a.svd_vjp(&a, &[1.], &a).is_err());
    let a = matrix(vec![f64::NAN], 1, 1);
    assert!(a.eigen_jvp(&a).is_err());
    let a = GraphMatrix::<f32>::from_row_major_data(vec![3.], 1, 1, None).unwrap();
    assert!(a.svd_jvp(&a).is_ok());
    let a = GraphMatrix::<i64>::from_row_major_data(vec![3], 1, 1, None).unwrap();
    assert!(a.eigen_jvp(&a).is_err());
}
