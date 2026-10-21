use groggy::storage::array::{numeric_reductions as k, NumArray};
#[test]
fn linear_rank_matches_independent_small_reference_and_preserves_source() {
    let source = vec![9.0, 1.0, 5.0, 3.0];
    let array = NumArray::new(source.clone());
    for (p, expected) in [(0.0, 1.0), (0.25, 2.5), (0.5, 4.0), (0.75, 6.0), (1.0, 9.0)] {
        assert_eq!(array.percentile(p), Some(expected));
        assert_eq!(k::percentile(&source, p), Some(expected));
    }
    assert_eq!(array.as_slice(), source);
    assert_eq!(array.describe().percentile_25, Some(2.5));
}
#[test]
fn extreme_midpoints_and_ieee_domains_are_explicit() {
    assert_eq!(k::percentile(&[i64::MIN, i64::MAX], 0.5), Some(-0.5));
    assert_eq!(k::percentile(&[-f64::MAX, f64::MAX], 0.5), Some(0.0));
    assert_eq!(k::percentile(&[f64::MAX, f64::MAX], 0.25), Some(f64::MAX));
    let interpolated = k::percentile(&[-f64::MAX, f64::MAX], 0.75).unwrap();
    assert!((interpolated / (f64::MAX * 0.5) - 1.0).abs() < 2e-15);
    assert_eq!(k::percentile(&[false, true], 0.25), Some(0.25));
    for p in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert_eq!(k::percentile(&[1.0], p), None);
    }
    assert_eq!(k::percentile(&[] as &[f64], 0.5), None);
    for p in [0.0, 0.25, 1.0] {
        assert!(k::percentile(&[1.0, f64::NAN], p).unwrap().is_nan());
        assert_eq!(k::percentile(&[7.0], p), Some(7.0));
    }
    assert_eq!(
        k::percentile(&[1.0, f64::INFINITY], 0.25),
        Some(f64::INFINITY)
    );
    assert!(k::percentile(&[f64::NEG_INFINITY, f64::INFINITY], 0.25)
        .unwrap()
        .is_nan());
    assert_eq!(
        k::percentile(&[f64::NEG_INFINITY, 1.0], 0.0),
        Some(f64::NEG_INFINITY)
    );
    for n in 1..35 {
        let values: Vec<_> = (0..n).map(|x| x as f64).rev().collect();
        for p in [0.0, 0.13, 0.5, 0.89, 1.0] {
            assert!((k::percentile(&values, p).unwrap() - p * (n - 1) as f64).abs() < 1e-13);
        }
    }
}
