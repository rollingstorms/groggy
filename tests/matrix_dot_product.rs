use groggy::storage::GraphMatrix;
fn dot(a: Vec<f64>, b: Vec<f64>) -> f64 {
    let n = a.len();
    GraphMatrix::from_row_major_data(a, 1, n, None)
        .unwrap()
        .multiply(&GraphMatrix::from_row_major_data(b, n, 1, None).unwrap())
        .unwrap()
        .get(0, 0)
        .unwrap()
}
#[test]
fn cancellation_survives_overflowing_products_and_distant_residuals() {
    assert_eq!(
        dot(vec![f64::MAX, f64::MAX, 1.], vec![f64::MAX, -f64::MAX, 1.]),
        1.
    );
    assert_eq!(dot(vec![1e100, 1., -1e100], vec![1., 1., 1.]), 1.);
    let tiny = f64::from_bits(1);
    assert_eq!(dot(vec![tiny, tiny], vec![0.5, 0.5]), tiny);
    assert_eq!(dot(vec![1., 2f64.powi(-53)], vec![1., 1.]), 1.);
    assert_eq!(
        dot(vec![1., 2f64.powi(-53), tiny], vec![1., 1., 1.]),
        f64::from_bits(1f64.to_bits() + 1)
    );
    assert_eq!(dot(vec![-tiny], vec![0.5]).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(dot(vec![f64::MAX], vec![2.]), f64::INFINITY);
    let a =
        GraphMatrix::<f32>::from_row_major_data(vec![f32::MAX, f32::MAX, 1.], 1, 3, None).unwrap();
    let b =
        GraphMatrix::<f32>::from_row_major_data(vec![f32::MAX, -f32::MAX, 1.], 3, 1, None).unwrap();
    assert_eq!(a.multiply(&b).unwrap().get(0, 0), Some(1.));
}
#[test]
fn nonfinite_inputs_follow_ieee_and_exact_cancellation_is_positive_zero() {
    assert!(dot(vec![0.], vec![f64::INFINITY]).is_nan());
    assert!(dot(vec![1., 1.], vec![f64::INFINITY, f64::NEG_INFINITY]).is_nan());
    assert!(dot(vec![f64::NAN], vec![1.]).is_nan());
    assert_eq!(dot(vec![2.], vec![f64::NEG_INFINITY]), f64::NEG_INFINITY);
    assert_eq!(
        dot(vec![1., -1.], vec![1., 1.]).to_bits(),
        0.0_f64.to_bits()
    );
}
