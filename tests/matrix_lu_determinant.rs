use groggy::storage::GraphMatrix;
fn matrix(v: Vec<f64>, n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(v, n, n, None).unwrap()
}
#[test]
fn pivoted_lu_reconstructs_scaled_and_singular_inputs_without_absolute_thresholds() {
    for scale in [1e-250, 1., 1e250] {
        let a = matrix(
            vec![
                0.,
                2. * scale,
                scale,
                3. * scale,
                scale,
                4. * scale,
                2. * scale,
                3. * scale,
                5. * scale,
            ],
            3,
        );
        let (p, l, u) = a.lu_decomposition().unwrap();
        let left = p.multiply(&a).unwrap();
        let right = l.multiply(&u).unwrap();
        for row in 0..3 {
            for col in 0..3 {
                assert!(
                    (left.get(row, col).unwrap() / scale - right.get(row, col).unwrap() / scale)
                        .abs()
                        < 2e-13
                );
                if row < col {
                    assert_eq!(l.get(row, col), Some(0.));
                }
                if row > col {
                    assert_eq!(u.get(row, col), Some(0.));
                }
            }
            assert_eq!(l.get(row, row), Some(1.));
        }
    }
    for a in [matrix(vec![0.; 9], 3), matrix(vec![0., 1., 0., 2.], 2)] {
        let (p, l, u) = a.lu_decomposition().unwrap();
        let left = p.multiply(&a).unwrap();
        let right = l.multiply(&u).unwrap();
        for row in 0..a.shape().0 {
            for col in 0..a.shape().1 {
                assert_eq!(left.get(row, col), right.get(row, col));
            }
        }
        assert_eq!(a.slogdet().unwrap(), (0., f64::NEG_INFINITY));
    }
}
#[test]
fn determinant_log_range_empty_sign_and_domain_behavior_are_explicit() {
    assert_eq!(matrix(vec![], 0).determinant().unwrap(), 1.);
    assert_eq!(matrix(vec![], 0).slogdet().unwrap(), (1., 0.));
    assert_eq!(matrix(vec![1., 2., 3., 4.], 2).determinant().unwrap(), -2.);
    let a = matrix(vec![1e300, 0., 0., 0., 1e-300, 0., 0., 0., 2.], 3);
    assert!((a.determinant().unwrap() - 2.).abs() < 1e-12);
    let a = matrix(vec![1e300, 0., 0., 0., 1e300, 0., 0., 0., 1e300], 3);
    assert!(a.determinant().unwrap().is_infinite());
    assert!(a.slogdet().unwrap().1.is_finite());
    assert_eq!(matrix(vec![f64::MAX; 4], 2).determinant().unwrap(), 0.);
    for value in [f64::NAN, f64::INFINITY] {
        assert!(matrix(vec![value], 1).determinant().is_err());
        assert!(matrix(vec![value], 1).lu_decomposition().is_err());
    }
    assert!(GraphMatrix::<i64>::from_row_major_data(vec![1], 1, 1, None)
        .unwrap()
        .determinant()
        .is_err());
}
