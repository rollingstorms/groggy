use groggy::storage::GraphMatrix;
#[test]
fn cholesky_reconstructs_spd_inputs_across_scale_and_supports_f32_and_empty() {
    for scale in [1e-300_f64, 1., 1e300] {
        let a = GraphMatrix::from_row_major_data(
            vec![4. * scale, 2. * scale, 2. * scale, 3. * scale],
            2,
            2,
            None,
        )
        .unwrap();
        let lower = a.cholesky_decomposition().unwrap();
        assert_eq!(lower.get(0, 1), Some(0.));
        let product = lower.multiply(&lower.transpose().unwrap()).unwrap();
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (product.get(row, col).unwrap() / scale - a.get(row, col).unwrap() / scale)
                        .abs()
                        < 1e-13
                );
            }
        }
        assert_eq!(a.get(0, 0), Some(4. * scale));
    }
    let a = GraphMatrix::<f32>::from_row_major_data(vec![4., 2., 2., 3.], 2, 2, None).unwrap();
    let lower = a.cholesky_decomposition().unwrap();
    assert!(
        (lower
            .multiply(&lower.transpose().unwrap())
            .unwrap()
            .get(1, 1)
            .unwrap()
            - 3.)
            .abs()
            < 1e-6
    );
    assert_eq!(
        GraphMatrix::<f64>::zeros(0, 0)
            .cholesky_decomposition()
            .unwrap()
            .shape(),
        (0, 0)
    );
}
#[test]
fn cholesky_rejects_asymmetric_indefinite_nonfinite_integer_and_rectangular_domains() {
    for values in [
        vec![1., 2., 0., 1.],
        vec![1., 2., 2., 1.],
        vec![0.; 4],
        vec![f64::NAN, 0., 0., 1.],
        vec![f64::INFINITY, 0., 0., 1.],
    ] {
        assert!(GraphMatrix::from_row_major_data(values, 2, 2, None)
            .unwrap()
            .cholesky_decomposition()
            .is_err());
    }
    assert!(GraphMatrix::<i64>::from_row_major_data(vec![1], 1, 1, None)
        .unwrap()
        .cholesky_decomposition()
        .is_err());
    assert!(GraphMatrix::from_row_major_data(vec![1., 2.], 1, 2, None)
        .unwrap()
        .cholesky_decomposition()
        .is_err());
}
