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

#[test]
fn exact_integer_fast_path_falls_back_without_losing_bits_or_nonfinite_precedence() {
    let large = 2f64.powi(63);
    let mut a = vec![large; 9];
    a.extend(vec![large; 9]);
    a.push(1.);
    let mut b = vec![large; 9];
    b.extend(vec![-large; 9]);
    b.push(1.);
    assert_eq!(dot(a, b), 1.);
    assert_eq!(dot(vec![large, large, 0.5], vec![large, -large, 1.]), 0.5);
    assert!(dot(vec![1., 0.], vec![1., f64::INFINITY]).is_nan());
    assert_eq!(dot(vec![2., 3.], vec![4., 5.]), 23.);
    for exponent in [-1000, -47, 17, 1000] {
        let base = 2f64.powi(exponent);
        let half_exponent = exponent - 53;
        let half = if half_exponent < -1022 {
            f64::from_bits(1_u64 << (half_exponent + 1074))
        } else {
            f64::from_bits(((half_exponent + 1023) as u64) << 52)
        };
        assert_eq!(dot(vec![base, half], vec![1., 1.]), base);
        assert_eq!(
            dot(vec![base, half, f64::from_bits(1)], vec![1., 1., 1.]),
            f64::from_bits(base.to_bits() + 1)
        );
    }
}
