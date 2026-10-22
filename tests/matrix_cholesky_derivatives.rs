use groggy::storage::GraphMatrix;
fn matrix(a: Vec<f64>, n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(a, n, n, None).unwrap()
}
#[test]
fn cholesky_derivatives_preserve_scales_identity_and_symmetric_pair_convention() {
    for scale in [1e-300, 1., 1e300] {
        let mut a = matrix(vec![4. * scale, 0., 0., 9. * scale], 2);
        a.set_row_labels(Some(vec!["r0".into(), "r1".into()]));
        a.set_column_names(vec!["c0".into(), "c1".into()]);
        let h = matrix(vec![2. * scale, 3. * scale, 3. * scale, 4. * scale], 2);
        let jvp = a.cholesky_jvp(&h).unwrap();
        let expected = [0.5, 0., 1.5, 2. / 3.];
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (jvp.get(r, c).unwrap() / scale.sqrt() - expected[r * 2 + c]).abs() < 1e-13
                );
            }
        }
        let g = matrix(vec![2., 100., 3., 4.], 2);
        let vjp = a.cholesky_vjp(&g).unwrap();
        let expected = [0.5, 0.75, 0.75, 2. / 3.];
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (vjp.get(r, c).unwrap() * scale.sqrt() - expected[r * 2 + c]).abs() < 1e-13
                );
            }
        }
        assert_eq!(vjp.row_labels(), a.row_labels());
        assert_eq!(vjp.column_names(), a.column_names());
        assert_eq!(jvp.column_names(), vec!["chol_0", "chol_1"]);
    }
    let a = matrix(vec![1e-300], 1);
    let v = a
        .cholesky_vjp(&matrix(vec![1e-320], 1))
        .unwrap()
        .get(0, 0)
        .unwrap();
    assert!((v / (1e-320 / 1e-150) - 0.5).abs() < 1e-14);
}
#[test]
fn rejects_invalid_derivative_domains_and_supports_empty_and_float32() {
    let a = matrix(vec![4., 0., 0., 4.], 2);
    let asymmetric = matrix(vec![1., 1., 0., 1.], 2);
    assert!(a.cholesky_jvp(&asymmetric).is_err());
    assert!(a.cholesky_vjp(&matrix(vec![1.], 1)).is_err());
    assert!(matrix(vec![0.], 1)
        .cholesky_vjp(&matrix(vec![1.], 1))
        .is_err());
    assert!(matrix(vec![1e-300], 1)
        .cholesky_vjp(&matrix(vec![1e300], 1))
        .is_err());
    for v in [f64::NAN, f64::INFINITY] {
        assert!(matrix(vec![1.], 1)
            .cholesky_jvp(&matrix(vec![v], 1))
            .is_err());
    }
    assert_eq!(
        matrix(vec![], 0)
            .cholesky_vjp(&matrix(vec![], 0))
            .unwrap()
            .shape(),
        (0, 0)
    );
    let a = GraphMatrix::<f32>::from_row_major_data(vec![4.], 1, 1, None).unwrap();
    assert_eq!(a.cholesky_vjp(&a).unwrap().get(0, 0), Some(1.));
    let a = GraphMatrix::<i64>::from_row_major_data(vec![4], 1, 1, None).unwrap();
    assert!(a.cholesky_vjp(&a).is_err());
}
